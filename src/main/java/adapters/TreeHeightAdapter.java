package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.List;

public class TreeHeightAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private Tree tree;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
    }

    @Override
    public int getNumImmutable() {
        return 0;
    }

    public int getNumMutable() {
        return 1;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[]{Math.log(this.tree.getRoot().getHeight())};
    }

    @Override
    public void update(double[] mutable, int nodeId) {
        double newHeight = Math.exp(mutable[0]);
        double scale = newHeight / this.tree.getRoot().getHeight();
        this.tree.scale(scale);
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return -getScaledNodeCount() * Math.log(this.tree.getRoot().getHeight());
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    private int getScaledNodeCount() {
        int count = 0;

        for (Node node : this.tree.getInternalNodes()) {
            if (!node.isFake() && (node.isRoot() || node.getParent().getHeight() != node.getHeight())) {
                count++;
            }
        }

        return count;
    }

}
