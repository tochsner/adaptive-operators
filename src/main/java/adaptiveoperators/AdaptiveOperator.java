package adaptiveoperators;

import adapters.Adapter;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.exception.MathIllegalArgumentException;

import java.util.ArrayList;
import java.util.List;

public class AdaptiveOperator extends Operator {

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");

    private List<Adapter> adapters;
    private Tree tree;
    private ConditionalSampler sampler;

    private final int burnIn = 2_000;
    private final int startTraining = 20_000;
    private final int endTraining = 100_000;
    private int count = 0;

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

        this.sampler = new MultivariateNormalSampler(totalNumImmutable, totalNumMutable);
    }

    @Override
    public double proposal() {
        // try {
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
                this.sampler.record(oldImmutable, oldMutable);
            }

            if (this.count < this.startTraining) {
                // we are in training phase
                // we don't change the state
                return 0;
            } else if (this.count == this.startTraining) {
                System.out.println("Start with adaptive kernel");
            } else if (this.count == this.endTraining) {
                System.out.println("End with adaptive kernel");
            }

            // sample from the conditional distribution

            double[] proposal;
            try {
                proposal = this.sampler.sampleConditionally(this.getImmutable(nodeId));
            } catch (MathIllegalArgumentException e) {
                return Double.NEGATIVE_INFINITY;
            }

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
            logDensityOld += this.sampler.logDensity(newImmutable, oldMutable);
            logDensityNew += this.sampler.logDensity(oldImmutable, proposal);

            return logDensityOld - logDensityNew;
//        } catch (Exception e) {
//            System.out.println(e);
//            return Double.NEGATIVE_INFINITY;
//        }
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

    private double[] getImmutable(int nodeId) {
        double[] immutable = new double[this.sampler.numConditions];

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
}
