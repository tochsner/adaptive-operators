package irreversible;

import adapters.Adapter;
import adapters.AdapterGenerator;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

public class GuidedRandomWalkOperator extends Operator {

    private static final double TARGET_ACCEPTANCE_PROBABILITY = 0.75;
    private static final double ADAPTATION_OFFSET = 10.0;
    private static final double ADAPTATION_EXPONENT = 0.6;
    private static final double MIN_SCALE_FACTOR = 1.0E-12;
    private static final double MAX_SCALE_FACTOR = 5.0;

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<List<AdapterGenerator>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");

    private List<Adapter> adapters;
    private List<AdapterGenerator> adapterGenerators;
    private Tree tree;
    private int totalNumMutable;

    private double[] scaleFactors;
    private long[] scaleFactorUpdateCount;
    private int[] directions;
    private int selectedParamIdx = -1;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();
        this.adapterGenerators = this.adapterGeneratorsInput.get();
        this.tree = this.treeInput.get();

        for (AdapterGenerator adapterGenerator : this.adapterGenerators) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        this.totalNumMutable = 0;
        for (Adapter adapter : this.adapters) {
            this.totalNumMutable += adapter.getNumMutable();
        }

        if (this.totalNumMutable == 0) {
            throw new IllegalArgumentException("GuidedRandomWalkOperator requires at least one mutable value");
        }

        this.scaleFactors = new double[this.totalNumMutable];
        this.scaleFactorUpdateCount = new long[this.totalNumMutable];
        this.directions = new int[this.totalNumMutable];

        for (int i = 0; i < this.totalNumMutable; i++) {
            this.scaleFactors[i] = 0.1;
            this.directions[i] = 1;
        }
    }

    @Override
    public double proposal() {
        this.selectedParamIdx = -1;

        try {
            int nodeId = this.chooseNodeId();
            ParameterSelection selection = this.chooseParameter();
            this.selectedParamIdx = selection.globalParamIdx;
            double oldLogJacobian = selection.adapter.getLogJacobianCorrection(nodeId);

            double[] mutable = selection.adapter.getMutable(nodeId);
            double current = mutable[selection.adapterParamIdx];
            double step = Math.abs(Randomizer.nextGaussian() * this.scaleFactors[selection.globalParamIdx]);
            mutable[selection.adapterParamIdx] = current + step * this.directions[selection.globalParamIdx];

            if (!Double.isFinite(mutable[selection.adapterParamIdx]) || !Double.isFinite(4*Math.exp(mutable[selection.adapterParamIdx]))) {
                return Double.NEGATIVE_INFINITY;
            }

            double transitionCorrection = selection.adapter.update(mutable, nodeId);
            double newLogJacobian = selection.adapter.getLogJacobianCorrection(nodeId);

            return oldLogJacobian - newLogJacobian + transitionCorrection;
        } catch (RuntimeException e) {
            e.printStackTrace();
            return Double.NEGATIVE_INFINITY;
        }
    }

    @Override
    public void reject(final int reason) {
        super.reject(reason);

        if (this.selectedParamIdx >= 0) {
            this.directions[this.selectedParamIdx] *= -1;
        }
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }

        return stateNodes;
    }

    @Override
    public void optimize(double logAlpha) {
        if (this.selectedParamIdx < 0) return;

        int paramIdx = this.selectedParamIdx;
        this.scaleFactorUpdateCount[paramIdx]++;

        double alpha = Math.exp(Math.min(logAlpha, 0.0));
        double learningRate = Math.pow(this.scaleFactorUpdateCount[paramIdx] + ADAPTATION_OFFSET, -ADAPTATION_EXPONENT);
        double logScaleFactor = Math.log(this.scaleFactors[paramIdx])
                + learningRate * (alpha - TARGET_ACCEPTANCE_PROBABILITY);
        double scaleFactor = Math.exp(logScaleFactor);

        this.scaleFactors[paramIdx] = Math.min(MAX_SCALE_FACTOR, Math.max(MIN_SCALE_FACTOR, scaleFactor));
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return TARGET_ACCEPTANCE_PROBABILITY;
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

    private ParameterSelection chooseParameter() {
        int globalParamIdx = Randomizer.nextInt(this.totalNumMutable);

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            int numMutable = adapter.getNumMutable();

            if (globalParamIdx < idx + numMutable) {
                return new ParameterSelection(adapter, globalParamIdx - idx, globalParamIdx);
            }

            idx += numMutable;
        }

        throw new IllegalStateException("Could not map parameter index " + globalParamIdx);
    }

    private static final class ParameterSelection {
        private final Adapter adapter;
        private final int adapterParamIdx;
        private final int globalParamIdx;

        private ParameterSelection(Adapter adapter, int adapterParamIdx, int globalParamIdx) {
            this.adapter = adapter;
            this.adapterParamIdx = adapterParamIdx;
            this.globalParamIdx = globalParamIdx;
        }
    }

}
