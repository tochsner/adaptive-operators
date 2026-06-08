package slice;

import adapters.Adapter;
import adapters.AdapterGenerator;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class LinCombSliceOperator extends SliceOperator {

    private static final double BURN_IN = 10_000;
    private static final double NUM_RECENT_SAMPLES = 10_000;

    private static final double TARGET_STEP_OUT_COUNT = 1.0;
    private static final double TARGET_SHRINK_COUNT = 1.0;
    private static final double ADAPTATION_OFFSET = 10.0;
    private static final double ADAPTATION_EXPONENT = 0.6;
    private static final double MIN_WINDOW_SIZE = 1.0E-12;
    private static final double MAX_WINDOW_SIZE = 1.0E12;

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<List<AdapterGenerator>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");

    private List<Adapter> adapters;
    private List<AdapterGenerator> adapterGenerators;
    private Tree tree;
    private int totalNumMutable;
    private LinkedList<double[]> mostRecentSamples;

    private double initialWindowSize = 1.0;
    private long windowUpdateCount;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();
        this.adapterGenerators = this.adapterGeneratorsInput.get();
        this.tree = this.treeInput.get();

        this.mostRecentSamples = new LinkedList<>();

        for (AdapterGenerator adapterGenerator : this.adapterGenerators) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        this.totalNumMutable = 0;
        for (Adapter adapter : this.adapters) {
            this.totalNumMutable += adapter.getNumMutable();
        }

        if (this.totalNumMutable == 0) {
            throw new IllegalArgumentException("StepOutShrinkSliceOperator requires at least one mutable value");
        }
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood) {
        // choose mutable parameter to work on

        int nodeId = this.chooseNodeId();

        double[] currentState = new double[this.totalNumMutable];
        int idx = 0;
        for (Adapter adapter : this.adapters) {
            System.arraycopy(
                    adapter.getMutable(nodeId), 0, currentState, idx, adapter.getNumMutable()
            );
            idx += adapter.getNumMutable();
        }

        this.remember(currentState);
        if (this.mostRecentSamples.size() < BURN_IN) return Double.NEGATIVE_INFINITY;

        final double[] otherStateA = this.mostRecentSamples.get(
                Randomizer.nextInt(this.mostRecentSamples.size())
        );
        final double[] otherStateB = this.mostRecentSamples.get(
                Randomizer.nextInt(this.mostRecentSamples.size())
        );

        final double currentLogLikelihood = computeCurrentLogLikelihood.get();
        final double logSliceDensity = currentLogLikelihood + Math.log(Randomizer.nextDouble());

        int finalNodeId = nodeId;
        Function<Double, Double> evaluateAt = (x) -> {
            double[] proposal = new double[this.totalNumMutable];

            for (int i = 0; i < this.totalNumMutable; i++) {
                proposal[i] = currentState[i] + x * (otherStateA[i] - otherStateB[i]);
            }

            try {
                this.updateAdapters(proposal, finalNodeId);
            } catch (RuntimeException e) {
                return Double.NEGATIVE_INFINITY;
            }

            return computeCurrentLogLikelihood.get();
        };

        Function<Double, Boolean> isIn = (x) -> evaluateAt.apply(x) >= logSliceDensity;

        // set up the initial window

        double windowSize = this.initialWindowSize;
        double x = 0;
        double l = x - windowSize * Randomizer.nextDouble();
        double r = l + windowSize;
        int stepOutCount = 0;
        int shrinkCount = 0;

        // expand lower limit if needed

        while (isIn.apply(l) && stepOutCount < 10) {
            l -= windowSize;
            stepOutCount++;
        }

        // expand upper limit if needed

        while (isIn.apply(r) && stepOutCount < 10) {
            r += windowSize;
            stepOutCount++;
        }

        // sample from current slice

        double newX = l + Randomizer.nextDouble() * (r - l);
        double newLogDensity = evaluateAt.apply(newX);

        while (newLogDensity < logSliceDensity) {
            // shrink

            shrinkCount++;

            if (newX < x) {
                l = newX;
            } else {
                r = newX;
            }

            newX = l + Randomizer.nextDouble() * (r - l);
            newLogDensity = evaluateAt.apply(newX);
        }

        this.updateWindowSize(stepOutCount, shrinkCount);

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

    private void updateWindowSize(int stepOutCount, int shrinkCount) {
        this.windowUpdateCount++;

        double learningRate = Math.pow(this.windowUpdateCount + ADAPTATION_OFFSET, -ADAPTATION_EXPONENT);
        double error = (stepOutCount - TARGET_STEP_OUT_COUNT) - (shrinkCount - TARGET_SHRINK_COUNT);
        double logWindowSize = Math.log(this.initialWindowSize) + learningRate * error;
        double windowSize = Math.exp(logWindowSize);

        this.initialWindowSize = Math.min(MAX_WINDOW_SIZE, Math.max(MIN_WINDOW_SIZE, windowSize));
    }

    private void remember(double[] sample) {
        this.mostRecentSamples.add(sample);

        while (NUM_RECENT_SAMPLES <= this.mostRecentSamples.size()) {
            this.mostRecentSamples.removeFirst();
        }
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
