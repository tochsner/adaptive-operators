package adaptiveoperators;

import adapters.Adapter;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AdaptiveOperator extends Operator {

    private static final double MIN_SCALE_FACTOR = 1.0E-3;
    private static final double MAX_SCALE_FACTOR = 1.0E3;

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Boolean> globalRedrawInput = new Input<>("globalRedraw",
            "If true, all mutable parameters of a node are redrawn from their learned conditional distribution " +
                    "given the immutable ones. If false, all mutable parameters are changed locally by adding " +
                    "zero-mean noise with the learned covariance to their current values.", true);
    public final Input<String> samplerInput = new Input<>("sampler",
            "The conditional sampler learning the distribution of the mutable parameters: multivariate_normal, " +
                    "centered_multivariate_normal, gaussian_mixture or neural_gaussian_mixture. Defaults to " +
                    "multivariate_normal for the global redraw and centered_multivariate_normal otherwise. The " +
                    "local random walk requires centered_multivariate_normal, as it relies on symmetric zero-mean noise.");

    private List<Adapter> adapters;
    private Tree tree;
    private ConditionalSampler sampler;
    private boolean globalRedraw;
    private int numMutable;
    private int numImmutable;

    private final int burnIn = 1_000;
    private final int startTraining = 5_000;
    private final int endTraining = 400_000;
    private int count = 0;

    private double scaleFactor = 1.0;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();
        this.tree = this.treeInput.get();

        int totalNumMutable = 0;
        int totalNumImmutable = 0;

        for (Adapter adapter : this.adapters) {
            totalNumMutable += adapter.getNumMutable();
            totalNumImmutable += adapter.getNumImmutable();
        }

        this.globalRedraw = this.globalRedrawInput.get();
        this.numMutable = totalNumMutable;
        this.numImmutable = totalNumImmutable;

        this.sampler = this.createSampler();
    }

    private ConditionalSampler createSampler() {
        SamplerType samplerType;
        if (this.samplerInput.get() != null) {
            samplerType = SamplerType.fromName(this.samplerInput.get());
        } else if (this.globalRedraw) {
            samplerType = SamplerType.MULTIVARIATE_NORMAL;
        } else {
            samplerType = SamplerType.CENTERED_MULTIVARIATE_NORMAL;
        }

        if (!this.globalRedraw && samplerType != SamplerType.CENTERED_MULTIVARIATE_NORMAL) {
            throw new IllegalArgumentException("The local random walk (globalRedraw=false) requires the " +
                    "centered_multivariate_normal sampler, but got '" + this.samplerInput.get() + "'");
        }

        // the local random walk learns the noise without conditioning on the immutable parameters
        int numConditions = this.globalRedraw ? this.numImmutable : 0;

        return switch (samplerType) {
            case MULTIVARIATE_NORMAL -> new MultivariateNormalSampler(numConditions, this.numMutable);
            case CENTERED_MULTIVARIATE_NORMAL -> new CenteredMultivariateNormalSampler(numConditions, this.numMutable);
            case GAUSSIAN_MIXTURE -> new GaussianMixtureSampler(numConditions, this.numMutable);
            case NEURAL_GAUSSIAN_MIXTURE -> new NeuralGaussianMixtureSampler(numConditions, this.numMutable);
        };
    }

    @Override
    public double proposal() {
        this.count++;

        if (this.count < this.burnIn) {
            // we are in burn in phase
            // we don't change the state nor record the state
            return 0;
        }

        int nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        while (this.tree.getNode(nodeId).isLeaf() || this.tree.getNode(nodeId).isRoot()) {
            nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        }

        double[] oldMutable = this.getMutable(nodeId);
        double[] oldImmutable = this.getImmutable(nodeId);

        // record the state in the sampler

        if (this.count < this.endTraining) {
            if (this.globalRedraw) {
                this.sampler.record(oldImmutable, oldMutable);
            } else {
                this.sampler.record(new double[0], oldMutable);
            }
        }

        if (this.count < this.startTraining) {
            // we are in the initial training phase
            // we don't change the state
            return 0;
        } else if (this.count == this.startTraining) {
            System.out.println("Start with adaptive kernel");
        } else if (this.count == this.endTraining) {
            System.out.println("End with adaptive kernel");
        }

        // sample from the conditional distribution

        double[] proposal = this.sampleProposal(oldMutable, oldImmutable);

        // update the adapters

        double logDensityOld = 0.0;
        double logDensityNew = 0.0;

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;

            logDensityOld += adapter.getLogJacobianCorrection(nodeId);

            double[] proposedMutable = new double[adapter.getNumMutable()];
            System.arraycopy(proposal, idx, proposedMutable, 0, adapter.getNumMutable());
            adapter.update(proposedMutable, nodeId);

            logDensityNew += adapter.getLogJacobianCorrection(nodeId);

            idx += adapter.getNumMutable();
        }

        // compute and return the log hastings ratio

        double[] newImmutable = this.getImmutable(nodeId);
        if (!Arrays.stream(newImmutable).allMatch(Double::isFinite)) {
            // the proposal led to an invalid state, so we reject it
            return Double.NEGATIVE_INFINITY;
        }

        if (this.globalRedraw) {
            // the local random walk is symmetric, so only the global redraw contributes proposal densities
            logDensityOld += this.sampler.logDensity(newImmutable, oldMutable, scaleFactor);
            logDensityNew += this.sampler.logDensity(oldImmutable, proposal, scaleFactor);
        }

        return logDensityOld - logDensityNew;
    }

    private double[] sampleProposal(double[] oldMutable, double[] oldImmutable) {
        if (this.globalRedraw) {
            return this.sampler.sampleConditionally(oldImmutable, this.scaleFactor);
        }

        // change all mutable parameters locally
        double[] noise = this.sampler.sampleConditionally(new double[0], this.scaleFactor);
        double[] proposal = new double[this.numMutable];
        for (int i = 0; i < this.numMutable; i++) {
            proposal[i] = oldMutable[i] + noise[i];
        }

        return proposal;
    }

    private double[] getMutable(int nodeId) {
        double[] mutable = new double[this.numMutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable(nodeId);
            System.arraycopy(adapterMutable, 0, mutable, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();
        }

        return mutable;
    }

    private double[] getImmutable(int nodeId) {
        double[] immutable = new double[this.numImmutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterImmutable = adapter.getImmutable(nodeId);
            System.arraycopy(adapterImmutable, 0, immutable, idx, adapter.getNumImmutable());
            idx += adapter.getNumImmutable();
        }

        return immutable;
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
        return this.scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("scaleFactor must be finite and positive");
        }
        this.scaleFactor = value;
    }

    @Override
    public void optimize(double logAlpha) {
        // the global redraw uses the learned conditional distribution as is
        if (this.globalRedraw) return;
        if (this.count < this.startTraining) return;

        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.scaleFactor);

        if (Double.isFinite(Math.exp(delta))) {
            this.scaleFactor = Math.min(MAX_SCALE_FACTOR, Math.max(MIN_SCALE_FACTOR, Math.exp(delta)));
        }
    }

    private enum SamplerType {
        MULTIVARIATE_NORMAL,
        CENTERED_MULTIVARIATE_NORMAL,
        GAUSSIAN_MIXTURE,
        NEURAL_GAUSSIAN_MIXTURE;

        private static SamplerType fromName(String name) {
            try {
                return SamplerType.valueOf(name.toUpperCase());
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException("Unknown sampler '" + name + "'", exception);
            }
        }
    }
}
