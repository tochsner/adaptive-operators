package mala;

import adapters.Adapter;
import adapters.AdapterGenerator;
import adaptiveoperators.AdaptiveOperator;
import adaptiveoperators.CenteredMultivariateNormalSampler;
import adaptiveoperators.ConditionalSamplerGradientModel;
import adaptiveoperators.GradientModel;
import adaptiveoperators.MultivariateNormalSampler;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import ml.MLP;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.RealMatrix;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;

/**
 * Adaptive, preconditioned MALA operator over the parameters attached to a tree node.
 *
 * <p>Each proposal takes a gradient-informed forward step and adds correlated Gaussian noise,
 * plugging into the shared {@link GaussianProposal} mechanics through an inner kernel. The
 * gradient of the (approximate) log target comes from a pluggable {@link GradientModel} —
 * either a neural network ({@code ml.MLP}) or a fitted multivariate normal wrapped by
 * {@link ConditionalSamplerGradientModel}, selected via {@code gradientModel}. The
 * preconditioner and the proposal noise come from a {@link CenteredMultivariateNormalSampler}
 * fitted to the visited mutable states.
 *
 * <p>The operator first runs a training phase (delegating proposals to the configured
 * alternative operators) so the gradient model and covariance can be learned from recorded
 * samples, then switches to its own gradient-based proposals.
 */
public class MALAOperator extends AdaptiveOperator {

    public final Input<List<AdapterGenerator<?>>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());
    public final Input<List<Operator>> alternativeOperatorsInput = new Input<>(
            "alternativeOperator",
            "",
            new ArrayList<>());
    public final Input<String> gradientModelInput = new Input<>(
            "gradientModel",
            "source of the approximate gradient: 'mlp' (neural network) or 'gaussian' (fitted multivariate normal)",
            "mlp");

    private Tree tree;

    protected List<Adapter> adapters;
    protected int totalNumMutable;
    protected int totalNumImmutable;

    protected List<Operator> alternativeOperators;

    private double alpha = 0.1;
    private CenteredMultivariateNormalSampler covarianceSampler;
    protected GradientModel gradientModel;

    protected int BURN_IN = 5_000;
    protected int JUST_TRAINING = 10_000;
    private double offset = 0.0;
    protected int count = 0;

    private Function<Double, Void> recordCallback = this::record;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.adapters = this.adaptersInput.get();
        this.alternativeOperators = this.alternativeOperatorsInput.get();

        for (AdapterGenerator<?> adapterGenerator : this.adapterGeneratorsInput.get()) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        for (Adapter adapter : this.adapters) {
            this.totalNumMutable += adapter.getNumMutable();
            this.totalNumImmutable += adapter.getNumImmutable();
        }

        this.covarianceSampler = new CenteredMultivariateNormalSampler(this.totalNumMutable);
        this.gradientModel = this.createGradientModel(this.totalNumMutable + this.totalNumImmutable);

        this.removeResults();
    }

    private GradientModel createGradientModel(int inputDimension) {
        String choice = this.gradientModelInput.get();

        if ("mlp".equalsIgnoreCase(choice)) {
            return new MLP(inputDimension, 1);
        }

        if ("gaussian".equalsIgnoreCase(choice)) {
            return new ConditionalSamplerGradientModel(new MultivariateNormalSampler(0, inputDimension));
        }

        throw new IllegalArgumentException("gradientModel must be 'mlp' or 'gaussian', got: " + choice);
    }

    @Override
    public double proposal() {
        if (this.count < this.JUST_TRAINING) {
            return this.proposeAlternativeOperator();
        }

        this.refreshAdapters();

        int nodeId = this.getNodeId();

        double[] oldState = this.getState(nodeId);
        double[] oldMutable = Arrays.copyOf(oldState, this.totalNumMutable);
        double[] immutable = Arrays.copyOfRange(oldState, this.totalNumMutable, oldState.length);

        GaussianProposalKernel kernel = new Kernel(immutable);

        return GaussianProposal.propose(
                oldMutable,
                kernel,
                proposed -> this.applyAdapters(proposed, nodeId)).logHastingsRatio();
    }

    /**
     * Applies the proposed mutable values through the adapters, accumulating the Jacobian and
     * transition corrections.
     */
    protected double applyAdapters(double[] proposedMutable, int nodeId) {
        double oldLogJacobian = 0.0;
        double newLogJacobian = 0.0;
        double transitionCorrection = 0.0;

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;

            oldLogJacobian += adapter.getLogJacobianCorrection(nodeId);

            double[] proposedSlice = new double[adapter.getNumMutable()];
            System.arraycopy(proposedMutable, idx, proposedSlice, 0, adapter.getNumMutable());

            try {
                transitionCorrection += adapter.update(proposedSlice, nodeId);
            } catch (Exception e) {
                return Double.NEGATIVE_INFINITY;
            }

            newLogJacobian += adapter.getLogJacobianCorrection(nodeId);

            idx += adapter.getNumMutable();
        }

        return oldLogJacobian - newLogJacobian + transitionCorrection;
    }

    protected int getNodeId() {
        int nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        while (this.tree.getNode(nodeId).isLeaf() || this.tree.getNode(nodeId).isRoot()) {
            nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        }
        return nodeId;
    }

    private void removeResults() {
        try {
            Files.deleteIfExists(Path.of("results.csv"));
        } catch (IOException e) {
            throw new IllegalStateException("failed to remove results.csv", e);
        }
    }

    protected double[] getState(int nodeId) {
        double[] oldMutable = this.getMutable(nodeId);
        double[] oldImmutable = this.getImmutable(nodeId);
        return assemble(oldMutable, oldImmutable);
    }

    public Function<Double, Void> getRecordCallback() {
        return this.recordCallback;
    }

    private Void record(double logLikelihood) {
        int nodeId = this.getNodeId();
        this.refreshAdapters();
        double[] state = this.getState(nodeId);
        this.record(state, logLikelihood);
        return null;
    }

    private void record(double[] state, double logLikelihood) {
        if (this.count++ < this.BURN_IN) return;
        if (this.count == this.JUST_TRAINING) System.out.println("Start to train approximate MALA.");

        if (this.offset == 0.0) this.offset = logLikelihood;

        this.gradientModel.record(state, new double[] {logLikelihood - this.offset});

        double[] mutableState = new double[this.totalNumMutable];
        System.arraycopy(state, 0, mutableState, 0, mutableState.length);
        this.covarianceSampler.record(new double[] {}, mutableState);
    }

    protected void refreshAdapters() {
        for (Adapter adapter : this.adapters) {
            adapter.refresh();
        }
    }

    private double[] getMutable(int nodeId) {
        double[] mutable = new double[this.totalNumMutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable(nodeId);
            System.arraycopy(adapterMutable, 0, mutable, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();
        }

        return mutable;
    }

    private double[] getImmutable(int nodeId) {
        double[] immutable = new double[this.totalNumImmutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterImmutable = adapter.getImmutable(nodeId);
            System.arraycopy(adapterImmutable, 0, immutable, idx, adapter.getNumImmutable());
            idx += adapter.getNumImmutable();
        }

        return immutable;
    }

    /**
     * Assembles a full network input [mutable, immutable] from a mutable vector and the
     * immutable context captured for the current proposal.
     */
    protected static double[] assemble(double[] mutable, double[] immutable) {
        double[] full = new double[mutable.length + immutable.length];
        System.arraycopy(mutable, 0, full, 0, mutable.length);
        System.arraycopy(immutable, 0, full, mutable.length, immutable.length);
        return full;
    }

    protected static boolean allFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) return false;
        }
        return true;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> nodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            nodes.addAll(adapter.listStateNodes());
        }

        return nodes;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.alpha;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("alpha must be finite and positive");
        }

        this.alpha = value;
    }

    protected double proposeAlternativeOperator() {
        return this.alternativeOperators.get(Randomizer.nextInt(this.alternativeOperators.size())).proposal();
    }

    @Override
    public void optimize(double logAlpha) {
        if (this.count < this.JUST_TRAINING) return;

        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.alpha);

        if (Double.isFinite(Math.exp(delta))) {
            this.alpha = Math.exp(delta);
        }
    }

    /**
     * Preconditioned MALA kernel: the drift is half the squared step times the learned
     * covariance applied to the gradient-model gradient, and the noise is drawn from that
     * same covariance.
     */
    private final class Kernel extends AbstractDensityKernel {

        private final double[] immutable;
        private final RealMatrix covariance;

        private Kernel(double[] immutable) {
            this.immutable = immutable;
            this.covariance = covarianceSampler.getCovariance();
        }

        @Override
        public double[] mean(double[] point) {
            double[] gradient = gradientModel.getGradient(assemble(point, this.immutable));
            double[] mutableGradient = Arrays.copyOf(gradient, totalNumMutable);
            double[] preconditioned = this.covariance.operate(new ArrayRealVector(mutableGradient)).toArray();

            double[] mean = new double[point.length];
            for (int i = 0; i < point.length; i++) {
                mean[i] = point[i] + 0.5 * alpha * alpha * Math.max(100, preconditioned[i]);
            }

            return mean;
        }

        @Override
        public double[] sampleNoise() {
            return covarianceSampler.sampleConditionally(new double[] {}, alpha);
        }

        @Override
        public double logDensity(double[] point, double[] mean) {
            double[] deviation = new double[point.length];
            for (int i = 0; i < point.length; i++) {
                deviation[i] = point[i] - mean[i];
            }

            return covarianceSampler.logDensity(new double[] {}, deviation, alpha);
        }
    }

}
