package adaptiveoperators;

import adapters.Adapter;
import adapters.AdapterGenerator;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.RealVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GuidedMixedPreconditionedCrankNicolsonOperator extends Operator {

    private static final double MIN_DELTA = 1.0E-12;
    private static final int MAX_DIRECTION_ATTEMPTS = 1_000;

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<List<AdapterGenerator>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private List<Adapter> adapters;
    private Tree tree;
    private CenteredMultivariateNormalSampler sampler;
    private int[] directions;
    private int selectedDirectionIdx = -1;

    private final int burnIn = 1_000;
    private final int startTraining = 5_000;
    private final int endTraining = 400_000;
    private int count = 0;

    private double scaleFactor = 1.0;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();
        this.tree = this.treeInput.get();

        for (AdapterGenerator adapterGenerator : this.adapterGeneratorsInput.get()) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        int totalNumMutable = 0;

        for (Adapter adapter : this.adapters) {
            totalNumMutable += adapter.getNumMutable();
        }

        if (totalNumMutable == 0) {
            throw new IllegalArgumentException("GuidedMixedPreconditionedCrankNicolsonOperator requires at least one mutable value");
        }

        this.sampler = new CenteredMultivariateNormalSampler(totalNumMutable);

        int numDirections = this.tree == null ? 1 : this.tree.getNodeCount();
        this.directions = new int[numDirections];
        Arrays.fill(this.directions, 1);
    }

    @Override
    public double proposal() {
        this.count++;
        this.selectedDirectionIdx = -1;

        if (this.count < this.burnIn) {
            // we are in burn in phase
            // we don't change the state nor record the state
            return 0;
        }

        int nodeId = this.chooseNodeId();
        this.selectedDirectionIdx = this.tree == null ? 0 : nodeId;

        this.refreshAdapters();
        double[] oldMutable = this.getMutable(nodeId);

        if (this.count < this.endTraining) {
            this.sampler.record(new double[] {}, oldMutable);
        }

        if (this.count < this.startTraining) {
            // we are in the initial training phase
            // we don't change the state
            return 0;
        } else if (this.count == this.startTraining) {
            System.out.println("Start with learning GMPCN kernel");
        } else if (this.count == this.endTraining) {
            System.out.println("End with learning GMPCN kernel");
        }

        double oldDelta = this.delta(oldMutable);
        if (oldDelta <= MIN_DELTA && this.directions[this.selectedDirectionIdx] < 0) {
            this.directions[this.selectedDirectionIdx] = 1;
        }

        double[] proposal = this.guidedProposal(oldMutable, oldDelta);
        if (proposal == null || !Arrays.stream(proposal).allMatch(Double::isFinite)) {
            return Double.NEGATIVE_INFINITY;
        }

        double logDensityOld = this.logReferenceDensity(oldDelta);
        double logDensityNew = this.logReferenceDensity(this.delta(proposal));
        double transitionCorrection = 0.0;

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            int numMutable = adapter.getNumMutable();
            if (numMutable == 0) continue;

            logDensityOld += adapter.getLogJacobianCorrection(nodeId);

            double[] mutable = new double[numMutable];
            System.arraycopy(proposal, idx, mutable, 0, numMutable);

            try {
                transitionCorrection += adapter.update(mutable, nodeId);
            } catch (Exception e) {
                return Double.NEGATIVE_INFINITY;
            }

            logDensityNew += adapter.getLogJacobianCorrection(nodeId);
            idx += numMutable;
        }

        return logDensityOld - logDensityNew + transitionCorrection;
    }

    @Override
    public void reject(final int reason) {
        super.reject(reason);

        if (this.selectedDirectionIdx >= 0) {
            this.directions[this.selectedDirectionIdx] *= -1;
        }
    }

    private double[] guidedProposal(double[] oldMutable, double oldDelta) {
        int direction = this.directions[this.selectedDirectionIdx];

        for (int attempt = 0; attempt < MAX_DIRECTION_ATTEMPTS; attempt++) {
            double gamma = this.sampleGamma(oldDelta);
            double[] proposal = this.mixedPcnProposal(oldMutable, gamma);
            double proposedDelta = this.delta(proposal);

            if ((proposedDelta - oldDelta) * direction > 0.0) {
                return proposal;
            }
        }

        return null;
    }

    private double[] mixedPcnProposal(double[] oldMutable, double gamma) {
        double[] center = this.sampler.mean;
        double[] perturbation = this.sampler.sampleConditionally(new double[] {}, this.scaleFactor / Math.sqrt(gamma));
        double shrinkage = Math.sqrt(1.0 - this.scaleFactor * this.scaleFactor);
        double[] proposal = new double[oldMutable.length];

        for (int i = 0; i < proposal.length; i++) {
            proposal[i] = center[i] + shrinkage * (oldMutable[i] - center[i]) + perturbation[i];
        }

        return proposal;
    }

    private double sampleGamma(double delta) {
        double shape = 0.5 * this.sampler.numValues;
        double rate = 0.5 * Math.max(delta, MIN_DELTA);
        return Randomizer.nextGamma(shape, rate);
    }

    double delta(double[] values) {
        double[] center = this.sampler.mean;
        RealVector diff = new ArrayRealVector(values).subtract(new ArrayRealVector(center));
        RealVector solved = this.sampler.choleskyDecomposition.getSolver().solve(diff);
        double delta = diff.dotProduct(solved);

        if (!Double.isFinite(delta)) {
            return Double.POSITIVE_INFINITY;
        }

        return Math.max(0.0, delta);
    }

    double logReferenceDensity(double delta) {
        return -0.5 * this.sampler.numValues * Math.log(Math.max(delta, MIN_DELTA));
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
        for (Adapter adapter : this.adapters) {
            adapter.refresh();
        }
    }

    private double[] getMutable(int nodeId) {
        double[] mutable = new double[this.sampler.numValues];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable(nodeId);
            System.arraycopy(adapterMutable, 0, mutable, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();
        }

        return mutable;
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
        if (!Double.isFinite(value) || value <= 0.0 || value > 1.0) {
            throw new IllegalArgumentException("scaleFactor must be finite and in (0, 1]");
        }
        this.scaleFactor = value;
    }

    @Override
    public void optimize(double logAlpha) {
        if (this.count < this.startTraining) return;

        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.scaleFactor);

        if (Double.isFinite(Math.exp(delta))) {
            this.scaleFactor = Math.min(0.999, Math.max(0.001, Math.exp(delta)));
        }
    }

}
