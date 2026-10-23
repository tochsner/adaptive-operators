package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.List;

/**
 * Immutable adapter that summarises a tree by the logarithm of its total branch length.
 */
public class TreeLengthAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private Tree tree;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
    }

    @Override
    public int getNumImmutable() {
        return 1;
    }

    @Override
    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[]{Math.log(this.computeTreeLength())};
    }

    private double computeTreeLength() {
        double length = 0.0;

        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                length += node.getLength();
            }
        }

        return length;
    }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double update(double[] mutable, int nodeId) {
        return 0.0;
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0.0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
