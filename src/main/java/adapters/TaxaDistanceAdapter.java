package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.List;

public class TaxaDistanceAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private Tree tree;
    public int taxaA;
    public int taxaB;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
    }

    @Override
    public int getNumImmutable() {
        return 3;
    }

    @Override
    public int getNumMutable() {
        return 1;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double[] getMutable(int nodeId) {

        return new double[]{0.0};
    }

    @Override
    public void update(double[] mutable, int nodeId) {

    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
