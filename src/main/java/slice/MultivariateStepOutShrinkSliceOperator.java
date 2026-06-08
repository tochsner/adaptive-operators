package slice;

import adapters.Adapter;
import adapters.AdapterGenerator;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

public class MultivariateStepOutShrinkSliceOperator extends SliceOperator {

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<List<AdapterGenerator>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");

    private List<Adapter> adapters;
    private List<AdapterGenerator> adapterGenerators;
    private Tree tree;
    private Random random;
    private int totalNumMutable;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();
        this.adapterGenerators = this.adapterGeneratorsInput.get();
        this.tree = this.treeInput.get();
        this.random = new Random(Randomizer.nextLong());

        for (AdapterGenerator adapterGenerator : this.adapterGenerators) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        this.totalNumMutable = 0;
        for (Adapter adapter : this.adapters) {
            this.totalNumMutable += adapter.getNumMutable();
        }

        if (this.totalNumMutable == 0) {
            throw new IllegalArgumentException("MultivariateStepOutShrinkSliceOperator requires at least one mutable value");
        }
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood) {
        // choose node and direction to work on

        int nodeId = this.chooseNodeId();

        final double[] x0 = this.getMutable(nodeId);
        final double[] direction = this.sampleDirection();

        final double currentLogLikelihood = computeCurrentLogLikelihood.get();
        final double logSliceDensity = currentLogLikelihood + Math.log(Randomizer.nextDouble());

        int finalNodeId = nodeId;
        Function<Double, Double> evaluateAt = (t) -> {
            // set values

            double[] proposal = new double[this.totalNumMutable];

            for (int i = 0; i < this.totalNumMutable; i++) {
                proposal[i] = x0[i] + t * direction[i];
            }

            try {
                this.updateAdapters(x0, finalNodeId);
                this.updateAdapters(proposal, finalNodeId);
            } catch (RuntimeException e) {
                return Double.NEGATIVE_INFINITY;
            }

            return computeCurrentLogLikelihood.get();
        };

        Function<Double, Boolean> isIn = (x) -> evaluateAt.apply(x) >= logSliceDensity;

        // set up the initial window

        double windowSize = 1.0;
        double x = 0.0;
        double l = x - windowSize * Randomizer.nextDouble();
        double r = l + windowSize;

        // expand lower limit if needed

        while (isIn.apply(l) && -20 < l) {
            l -= windowSize;
        }

        // expand upper limit if needed

        while (isIn.apply(r) && r < 20) {
            r += windowSize;
        }

        // sample from current slice

        double newX = l + Randomizer.nextDouble() * (r - l);
        double newLogDensity = evaluateAt.apply(newX);

        while (newLogDensity < logSliceDensity) {
            // shrink

            if (newX < x) {
                l = newX;
            } else {
                r = newX;
            }

            newX = l + Randomizer.nextDouble() * (r - l);
            newLogDensity = evaluateAt.apply(newX);
        }

        return currentLogLikelihood - newLogDensity;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }

        return stateNodes;
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

    private double[] sampleDirection() {
        double[] direction = new double[this.totalNumMutable];
        double normSquared = 0.0;

        for (int i = 0; i < this.totalNumMutable; i++) {
            direction[i] = this.random.nextGaussian();
            normSquared += direction[i] * direction[i];
        }

        double norm = Math.sqrt(normSquared);

        for (int i = 0; i < this.totalNumMutable; i++) {
            direction[i] /= norm;
        }

        return direction;
    }

    private void updateAdapters(double[] mutable, int nodeId) {
        int idx = 0;

        for (Adapter adapter : this.adapters) {
            int numMutable = adapter.getNumMutable();
            if (numMutable == 0) continue;

            double[] adapterMutable = new double[numMutable];
            System.arraycopy(mutable, idx, adapterMutable, 0, numMutable);
            adapter.update(adapterMutable, nodeId);

            idx += numMutable;
        }
    }

}
