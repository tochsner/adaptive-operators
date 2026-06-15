package adaptiveoperators;

import adapters.Adapter;
import adapters.AdapterGenerator;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PreconditionedCrankNicolsonOperator extends Operator {

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<List<AdapterGenerator>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private List<Adapter> adapters;
    private Tree tree;
    private CenteredMultivariateNormalSampler sampler;

    private final int burnIn = 100;
    private final int startTraining = 1_000;
    private final int endTraining = 400_000;
    private int count = 0;

    private double scaleFactor = 0.2;

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

        this.sampler = new CenteredMultivariateNormalSampler(totalNumMutable);
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

        this.refreshAdapters();
        double[] oldMutable = this.getMutable(nodeId);

        // record the state in the sampler

        if (this.count < this.endTraining) {
            this.sampler.record(new double[] {}, oldMutable);
        }

        if (this.count < this.startTraining) {
            // we are in the initial training phase
            // we don't change the state
            return 0;
        } else if (this.count == this.startTraining) {
            System.out.println("Start with learning pCN kernel");
        } else if (this.count == this.endTraining) {
            System.out.println("End with learning pCN kernel");
        }

        // sample from the conditional distribution

        double[] perturbation = this.sampler.sampleConditionally(new double[] {}, 1.0);

        // update the adapters

        double logDensityOld = 0.0;
        double logDensityNew = 0.0;
        double transitionCorrection = 0.0;
        double[] mean = this.sampler.getMean();
        double[] proposal = new double[oldMutable.length];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;

            logDensityOld += adapter.getLogJacobianCorrection(nodeId);

            double[] mutable = adapter.getMutable(nodeId);
            for (int i = 0; i < adapter.getNumMutable(); i++) {
                int coordinate = idx++;
                mutable[i] = mean[coordinate]
                        + Math.sqrt(1.0 - this.scaleFactor * this.scaleFactor) * (mutable[i] - mean[coordinate])
                        + this.scaleFactor * perturbation[coordinate];
                proposal[coordinate] = mutable[i];
            }

            try {
                transitionCorrection += adapter.update(mutable, nodeId);
            } catch (Exception e) {
                return Double.NEGATIVE_INFINITY;
            }

            logDensityNew += adapter.getLogJacobianCorrection(nodeId);
        }

        if (!Arrays.stream(proposal).allMatch(Double::isFinite)) {
            return Double.NEGATIVE_INFINITY;
        }

        logDensityOld += this.sampler.logDensity(new double[] {}, centered(oldMutable, mean), 1.0);
        logDensityNew += this.sampler.logDensity(new double[] {}, centered(proposal, mean), 1.0);

        return logDensityOld - logDensityNew + transitionCorrection;
    }

    private static double[] centered(double[] values, double[] mean) {
        double[] centered = new double[values.length];

        for (int i = 0; i < values.length; i++) {
            centered[i] = values[i] - mean[i];
        }

        return centered;
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
