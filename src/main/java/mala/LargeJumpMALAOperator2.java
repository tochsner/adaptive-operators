package mala;

import adapters.Adapter;
import adapters.AdapterGenerator;
import adaptiveoperators.CenteredMultivariateNormalSampler;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import slice.SliceOperator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Mode-jumping MALA-style operator with separate parameter groups for the jump and for the
 * optimization/noise. Its forward step builds a deterministic map from (1) a large jump on the
 * <em>jump</em> adapters' parameters and (2) a short random-walk optimization, driven by a given
 * set of operators keeping only likelihood-increasing moves, on the <em>optimization</em>
 * adapters' parameters; correlated Gaussian noise is then added to the optimization parameters.
 *
 * <p>Only the scalar jump scaling is learned (it is the coercible parameter, adapted towards the
 * target acceptance rate). The proposal noise is not separately tuned: it is drawn from a
 * {@link CenteredMultivariateNormalSampler} covariance fitted online to the visited optimization
 * parameters, and the same covariance defines the Hastings density. For the first {@code burnIn}
 * proposals the operator only records the optimization parameters and delegates to one of the
 * optimization operators, so the covariance is learned before it is used to drive a jump.
 *
 * <p>Each proposal conditions on the jump {@code J} and the optimization seed {@code S}. The jump
 * parameters move deterministically ({@code +J} forward, {@code −J} on the reverse, which returns
 * them to the start), so with a symmetric jump they contribute only their transform Jacobian. The
 * remaining randomness is the noise on the optimization parameters, leaving the Hastings ratio
 * {@code logN(x_opt; m_r) − logN(y_opt; m_f)} under the learned covariance — the same difference
 * {@link AbstractDensityKernel#logProposalRatio} computes, collapsing to noise-only when the
 * optimization converges.
 *
 * <p>The optimization evaluates the posterior mid-proposal, so the operator extends
 * {@link SliceOperator} and receives the likelihood supplier from {@code mcmc.SliceMCMC}, the same
 * mechanism {@link slice.StepOutShrinkSliceOperator} uses. The optimization operators are assumed
 * to mutate only the selected node's optimization-adapter parameters, so a rejected step is
 * reverted by re-applying the previous mutable vector through those adapters.
 */
public class LargeJumpMALAOperator2 extends SliceOperator {

    public final Input<List<Adapter>> jumpAdaptersInput = new Input<>(
            "jumpAdapter", "adapters whose parameters receive the large jump", new ArrayList<>());
    public final Input<List<AdapterGenerator>> jumpAdapterGeneratorsInput = new Input<>(
            "jumpAdapterGenerator", "", new ArrayList<>());
    public final Input<List<Adapter>> optimizationAdaptersInput = new Input<>(
            "optimizationAdapter", "adapters whose parameters are optimized and receive the noise", new ArrayList<>());
    public final Input<List<AdapterGenerator>> optimizationAdapterGeneratorsInput = new Input<>(
            "optimizationAdapterGenerator", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<List<Operator>> optimizationOperatorsInput = new Input<>(
            "optimizationOperator",
            "operators used for the likelihood-increasing random-walk optimization",
            new ArrayList<>());
    public final Input<Integer> numOptimizationStepsInput = new Input<>(
            "numOptimizationSteps",
            "number of random-walk steps in the optimization phase",
            100);
    public final Input<Double> jumpScaleInput = new Input<>(
            "jumpScale",
            "initial standard deviation of the isotropic large jump (learned)",
            1.0);
    public final Input<Double> noiseScaleInput = new Input<>(
            "noiseScale",
            "fixed scaling applied to the learned covariance when drawing the proposal noise",
            1.0);
    public final Input<Integer> burnInInput = new Input<>(
            "burnIn",
            "number of initial proposals that record the optimization parameters and delegate to an "
                    + "optimization operator while the covariance is learned",
            2_000);

    // package-private so same-package unit tests can drive the optimization with fakes
    List<Adapter> jumpAdapters;
    List<Adapter> optimizationAdapters;
    List<Operator> optimizationOperators;
    private Tree tree;
    int numJumpMutable;
    int numOptimizationMutable;
    int numOptimizationSteps;
    int burnIn;
    int count = 0;
    double jumpScale;
    double noiseScale;
    CenteredMultivariateNormalSampler jumpCovarianceSampler;
    CenteredMultivariateNormalSampler optimizationCovarianceSampler;

    @Override
    public void initAndValidate() {
        this.jumpAdapters = this.jumpAdaptersInput.get();
        this.optimizationAdapters = this.optimizationAdaptersInput.get();
        this.optimizationOperators = this.optimizationOperatorsInput.get();
        this.tree = this.treeInput.get();
        this.numOptimizationSteps = this.numOptimizationStepsInput.get();
        this.burnIn = this.burnInInput.get();
        this.jumpScale = this.jumpScaleInput.get();
        this.noiseScale = this.noiseScaleInput.get();

        for (AdapterGenerator adapterGenerator : this.jumpAdapterGeneratorsInput.get()) {
            this.jumpAdapters.addAll(adapterGenerator.getAdapters());
        }
        for (AdapterGenerator adapterGenerator : this.optimizationAdapterGeneratorsInput.get()) {
            this.optimizationAdapters.addAll(adapterGenerator.getAdapters());
        }

        this.numJumpMutable = countMutable(this.jumpAdapters);
        this.numOptimizationMutable = countMutable(this.optimizationAdapters);

        if (this.numJumpMutable == 0) {
            throw new IllegalArgumentException("LargeJumpMALAOperator requires at least one mutable jump parameter");
        }

        if (this.numOptimizationMutable == 0) {
            throw new IllegalArgumentException("LargeJumpMALAOperator requires at least one mutable optimization parameter");
        }

        if (this.optimizationOperators.isEmpty()) {
            throw new IllegalArgumentException("LargeJumpMALAOperator requires at least one optimizationOperator");
        }

        if (this.numOptimizationSteps < 0) {
            throw new IllegalArgumentException("numOptimizationSteps must be non-negative");
        }

        if (this.burnIn < 0) {
            throw new IllegalArgumentException("burnIn must be non-negative");
        }

        if (!Double.isFinite(this.jumpScale) || this.jumpScale <= 0.0) {
            throw new IllegalArgumentException("jumpScale must be finite and positive");
        }

        if (!Double.isFinite(this.noiseScale) || this.noiseScale <= 0.0) {
            throw new IllegalArgumentException("noiseScale must be finite and positive");
        }

        this.optimizationCovarianceSampler = new CenteredMultivariateNormalSampler(this.numOptimizationMutable);
        this.jumpCovarianceSampler = new CenteredMultivariateNormalSampler(this.numJumpMutable);
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood, State state) {
        int nodeId = this.chooseNodeId();
        this.refreshAdapters();

        double[] oldStateOpt = this.getMutableVector(this.optimizationAdapters, this.numOptimizationMutable, nodeId);
        this.optimizationCovarianceSampler.record(new double[] {}, oldStateOpt);

        double[] oldStateJump = this.getMutableVector(this.jumpAdapters, this.numJumpMutable, nodeId);
        this.jumpCovarianceSampler.record(new double[] {}, oldStateJump);

        // burn-in: keep the chain moving with a simple optimization move while the covariance is
        // still being learned, before driving any jump with it
        if (this.count++ < this.burnIn) {
            return this.optimizationOperators.get(Randomizer.nextInt(this.optimizationOperators.size())).proposal();
        }

        double oldLogJacobian = this.sumLogJacobianCorrection(this.jumpAdapters, nodeId)
                + this.sumLogJacobianCorrection(this.optimizationAdapters, nodeId);

        // draw the auxiliaries and the noise from the main stream before any reseed; these draws
        // also materialize this thread's Randomizer instance, so the forward and reverse
        // setSeed(seed) below both reseed the same RNG and the two walks are coupled

        double[] jump = this.jumpCovarianceSampler.sampleConditionally(new double[] {}, this.jumpScale);

        long seed = Randomizer.nextLong();
        double[] noise = this.optimizationCovarianceSampler.sampleConditionally(new double[] {}, this.noiseScale);
        long continuationSeed = Randomizer.nextLong();

        // forward: jump the jump parameters by +J, then optimize the optimization parameters

        double[] forwardMean = this.jumpAndOptimize(jump, oldStateOpt, seed, nodeId, computeCurrentLogLikelihood, state);
        if (forwardMean == null) return Double.NEGATIVE_INFINITY;

        double[] proposedJump = add(oldStateJump, jump);
        double[] proposedOptimization = new double[this.numOptimizationMutable];
        for (int i = 0; i < this.numOptimizationMutable; i++) {
            proposedOptimization[i] = forwardMean[i] + noise[i];
            if (!Double.isFinite(proposedOptimization[i])) return Double.NEGATIVE_INFINITY;
        }

        // apply the proposal y and record its Jacobian and transition correction

        double transitionCorrection = this.applyVector(this.jumpAdapters, proposedJump, nodeId)
                + this.applyVector(this.optimizationAdapters, proposedOptimization, nodeId);
        if (!Double.isFinite(transitionCorrection)) return Double.NEGATIVE_INFINITY;
        double newLogJacobian = this.sumLogJacobianCorrection(this.jumpAdapters, nodeId)
                + this.sumLogJacobianCorrection(this.optimizationAdapters, nodeId);

        // reverse: jump the jump parameters by -J (back to the start) and optimize again

        double[] reverseMean = this.jumpAndOptimize(negate(jump), proposedOptimization, seed, nodeId, computeCurrentLogLikelihood, state);
        if (reverseMean == null) return Double.NEGATIVE_INFINITY;

        // restore the state to the proposal y, then continue the main stream from a fresh seed

        if (!Double.isFinite(this.applyVector(this.jumpAdapters, proposedJump, nodeId))) return Double.NEGATIVE_INFINITY;
        if (!Double.isFinite(this.applyVector(this.optimizationAdapters, proposedOptimization, nodeId))) return Double.NEGATIVE_INFINITY;
        Randomizer.setSeed(continuationSeed);

        // Hastings ratio over the optimization parameters under the learned covariance

        double ratio = this.logDensity(oldStateOpt, reverseMean) - this.logDensity(proposedOptimization, forwardMean);
        if (!Double.isFinite(ratio)) return Double.NEGATIVE_INFINITY;

        return ratio + oldLogJacobian - newLogJacobian + transitionCorrection;
    }

    /**
     * Runs the conditioned forward/reverse map: seed the walk with {@code seed}, apply the jump
     * delta to the jump parameters, set the optimization parameters to {@code startOptimization},
     * then take {@code numOptimizationSteps} random-walk steps keeping only likelihood-increasing
     * moves. Returns the optimized optimization vector and leaves the state there, or {@code null}
     * if the starting point cannot be evaluated. Does not restore the RNG; the caller reseeds once
     * both walks have run.
     */
    double[] jumpAndOptimize(double[] jumpDelta, double[] startOptimization, long seed, int nodeId, Supplier<Double> ll, State state) {
        Randomizer.setSeed(seed);

        double[] jumpedJump = add(this.getMutableVector(this.jumpAdapters, this.numJumpMutable, nodeId), jumpDelta);
        if (!Double.isFinite(this.applyVector(this.jumpAdapters, jumpedJump, nodeId))) return null;
        if (!Double.isFinite(this.applyVector(this.optimizationAdapters, startOptimization, nodeId))) return null;

        double[] best = startOptimization;
        double bestLogLikelihood = ll.get();
        if (!Double.isFinite(bestLogLikelihood)) return null;

        for (int step = 0; step < this.numOptimizationSteps; step++) {
            Operator operator = this.optimizationOperators.get(Randomizer.nextInt(this.optimizationOperators.size()));

            double operatorLogHastings = operator.proposal();

            double candidateLogLikelihood;
            if (operatorLogHastings == Double.NEGATIVE_INFINITY) {
                candidateLogLikelihood = Double.NEGATIVE_INFINITY;
            } else {
                candidateLogLikelihood = ll.get();
            }

            if (candidateLogLikelihood > bestLogLikelihood) {
                best = this.getMutableVector(this.optimizationAdapters, this.numOptimizationMutable, nodeId);
                bestLogLikelihood = candidateLogLikelihood;
            } else {
                // revert
                this.applyVector(optimizationAdapters, best, nodeId);
            }
        }

        return best;
    }

    private int chooseNodeId() {
        if (this.tree == null) {
            return 0;
        }

        int nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        while (this.tree.getNode(nodeId).isLeaf() || this.tree.getNode(nodeId).isRoot()) {
            nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        }

        return nodeId;
    }

    private void refreshAdapters() {
        for (Adapter adapter : this.jumpAdapters) {
            adapter.refresh();
        }
        for (Adapter adapter : this.optimizationAdapters) {
            adapter.refresh();
        }
    }

    private double[] getMutableVector(List<Adapter> adapters, int numMutable, int nodeId) {
        double[] mutable = new double[numMutable];

        int idx = 0;
        for (Adapter adapter : adapters) {
            double[] adapterMutable = adapter.getMutable(nodeId);
            System.arraycopy(adapterMutable, 0, mutable, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();
        }

        return mutable;
    }

    /**
     * Applies a mutable vector across a group of adapters and returns the summed transition
     * correction, or {@link Double#NEGATIVE_INFINITY} if any adapter rejects the values.
     */
    private double applyVector(List<Adapter> adapters, double[] mutable, int nodeId) {
        double transitionCorrection = 0.0;

        int idx = 0;
        for (Adapter adapter : adapters) {
            int numMutable = adapter.getNumMutable();
            if (numMutable == 0) continue;

            double[] slice = Arrays.copyOfRange(mutable, idx, idx + numMutable);

            try {
                transitionCorrection += adapter.update(slice, nodeId);
            } catch (RuntimeException e) {
                return Double.NEGATIVE_INFINITY;
            }

            idx += numMutable;
        }

        return transitionCorrection;
    }

    private double sumLogJacobianCorrection(List<Adapter> adapters, int nodeId) {
        double logJacobian = 0.0;

        for (Adapter adapter : adapters) {
            if (adapter.getNumMutable() == 0) continue;
            logJacobian += adapter.getLogJacobianCorrection(nodeId);
        }

        return logJacobian;
    }

    /** Log density of the optimization point under the learned, noise-scaled covariance. */
    double logDensity(double[] point, double[] mean) {
        double[] deviation = new double[point.length];
        for (int i = 0; i < point.length; i++) {
            deviation[i] = point[i] - mean[i];
        }

        return this.optimizationCovarianceSampler.logDensity(new double[] {}, deviation, this.noiseScale);
    }

    private static int countMutable(List<Adapter> adapters) {
        int total = 0;
        for (Adapter adapter : adapters) {
            total += adapter.getNumMutable();
        }
        return total;
    }

    private static double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    private static double[] negate(double[] a) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = -a[i];
        }
        return result;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>();

        for (Adapter adapter : this.jumpAdapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }

        for (Adapter adapter : this.optimizationAdapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }

        for (Operator operator : this.optimizationOperators) {
            stateNodes.addAll(operator.listStateNodes());
        }

        return stateNodes;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.jumpScale;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("jumpScale must be finite and positive");
        }

        this.jumpScale = value;
    }

    @Override
    public void optimize(double logAlpha) {
//        double delta = this.calcDelta(logAlpha);
//        delta += Math.log(this.jumpScale);
//        this.jumpScale = Math.exp(delta);
//        if (Randomizer.nextDouble() < 0.01) System.out.println(jumpScale);
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return 0.01;
    }
}
