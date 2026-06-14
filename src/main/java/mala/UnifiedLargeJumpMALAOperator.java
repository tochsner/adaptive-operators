package mala;

import adapters.Adapter;
import adapters.AdapterGenerator;
import adaptiveoperators.CenteredMultivariateNormalSampler;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
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
public class UnifiedLargeJumpMALAOperator extends SliceOperator {

    public final Input<List<Adapter>> jumpAdaptersInput = new Input<>(
            "jumpAdapter", "adapters whose parameters receive the large jump", new ArrayList<>());
    public final Input<List<AdapterGenerator>> jumpAdapterGeneratorsInput = new Input<>(
            "jumpAdapterGenerator", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Integer> numOptimizationStepsInput = new Input<>(
            "numOptimizationSteps",
            "number of random-walk steps in the optimization phase",
            20);
    public final Input<Double> jumpScaleInput = new Input<>(
            "jumpScale",
            "initial standard deviation of the isotropic large jump (learned)",
            0.01);
    public final Input<Double> noiseScaleInput = new Input<>(
            "noiseScale",
            "fixed scaling applied to the learned covariance when drawing the proposal noise",
            0.001);
    public final Input<Integer> burnInInput = new Input<>(
            "burnIn",
            "number of initial proposals that record the optimization parameters and delegate to an "
                    + "optimization operator while the covariance is learned",
            500);

    List<Adapter> jumpAdapters;
    int numJumpMutable;

    private Tree tree;
    double jumpScale;
    double noiseScale;
    int numOptimizationSteps;

    CenteredMultivariateNormalSampler jumpCovarianceSampler;

    int burnIn;
    int count = 0;

    @Override
    public void initAndValidate() {
        this.jumpAdapters = this.jumpAdaptersInput.get();
        this.tree = this.treeInput.get();
        this.numOptimizationSteps = this.numOptimizationStepsInput.get();
        this.burnIn = this.burnInInput.get();
        this.jumpScale = this.jumpScaleInput.get();
        this.noiseScale = this.noiseScaleInput.get();

        for (AdapterGenerator adapterGenerator : this.jumpAdapterGeneratorsInput.get()) {
            this.jumpAdapters.addAll(adapterGenerator.getAdapters());
        }

        this.numJumpMutable = countMutable(this.jumpAdapters);
        this.jumpCovarianceSampler = new CenteredMultivariateNormalSampler(this.numJumpMutable);
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood, State state) {
        int nodeId = this.chooseNodeId();
        this.refreshAdapters();

        // record states

        double[] oldState = this.getMutableVector(this.jumpAdapters, this.numJumpMutable, nodeId);
        this.jumpCovarianceSampler.record(new double[] {}, oldState);

        this.count++;
        if (this.count < this.burnIn) {
            return Double.NEGATIVE_INFINITY;
        } else if (this.count == this.burnIn) {
            System.out.println("Large jumps start");
        }

        // record the old Jacobian

        double oldLogJacobian = this.sumLogJacobianCorrection(this.jumpAdapters, nodeId);

        // jump

        double[] jump = this.jumpCovarianceSampler.sampleConditionally(new double[] {}, this.jumpScale);
        double[] jumpedState = add(oldState, jump);
        this.applyVector(this.jumpAdapters, jumpedState, nodeId);

        // optimize

        double[] optimizedState = this.optimizeState(nodeId, computeCurrentLogLikelihood);

        // add noise

        double[] noise = this.jumpCovarianceSampler.sampleConditionally(new double[] {}, this.noiseScale);
        double[] proposedState = add(optimizedState, noise);
        this.applyVector(this.jumpAdapters, proposedState, nodeId);

        // record the new Jacobian

        double newLogJacobian = this.sumLogJacobianCorrection(this.jumpAdapters, nodeId);

        // reverse: jump the jump parameters by -J and optimize again

        double[] reverseJumpedState = add(proposedState, negate(jump));
        this.applyVector(this.jumpAdapters, reverseJumpedState, nodeId);

        double[] reverseOptimizedState = this.optimizeState(nodeId, computeCurrentLogLikelihood);

        // restore the state to the proposal

        this.applyVector(this.jumpAdapters, proposedState, nodeId);

        // Hastings ratio over the optimization parameters under the learned covariance

        double ratio = this.logDensity(oldState, reverseOptimizedState) - this.logDensity(proposedState, optimizedState);

        return ratio + oldLogJacobian - newLogJacobian;
    }

    private double[] optimizeState(int nodeId, Supplier<Double> computeCurrentLogLikelihood) {
        double[] best = this.getMutableVector(this.jumpAdapters, this.numJumpMutable, nodeId);
        double bestLogLikelihood = computeCurrentLogLikelihood.get();

        for (int step = 0; step < this.numOptimizationSteps; step++) {
            double[] noise = this.jumpCovarianceSampler.sampleConditionally(new double[] {}, 0.1 * this.noiseScale);
            double[] candidate = add(best, noise);
            this.applyVector(this.jumpAdapters, candidate, nodeId);

            double candidateLogLikelihood = computeCurrentLogLikelihood.get();

            if (candidateLogLikelihood > bestLogLikelihood) {
                best = candidate;
                bestLogLikelihood = candidateLogLikelihood;
            } else {
                this.applyVector(jumpAdapters, best, nodeId);
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
            transitionCorrection += adapter.update(slice, nodeId);

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

        return this.jumpCovarianceSampler.logDensity(new double[] {}, deviation, this.noiseScale);
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

        return stateNodes;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.noiseScale;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("noiseScale must be finite and positive");
        }

        this.noiseScale = value;
    }

    @Override
    public void optimize(double logAlpha) {
//        double delta = this.calcDelta(logAlpha);
//        delta += Math.log(this.noiseScale);
//        this.noiseScale = Math.exp(delta);
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return 0.05;
    }

    @Override
    public String getName() {
        String className = this.getClass().getName();
        return className + "(js " + this.jumpScale + ", ns " + this.noiseScale + ")";
    }
}
