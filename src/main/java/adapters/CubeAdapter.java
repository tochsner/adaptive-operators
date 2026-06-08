package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class CubeAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private Tree tree;
    private LinkedList<Integer> cube;
    private Random random;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.random = new Random();

        this.refresh();
    }

    @Override
    public int getNumImmutable() {
        return 0;
    }

    @Override
    public int getNumMutable() {
        return this.tree.getLeafNodeCount() - 1;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double[] getMutable(int nodeId) {
        double[] cubeDistances = TreeUtils.getCubeDistances(this.tree, this.cube);

        for (int i = 0; i < cubeDistances.length; i++) {
            cubeDistances[i] = Math.log(cubeDistances[i]);
        }

        return cubeDistances;
    }

    @Override
    public double update(double[] mutable, int nodeId) {
        double[] cubeDistances = new double[this.getNumMutable()];

        for (int i = 0; i < cubeDistances.length; i++) {
            cubeDistances[i] = Math.exp(mutable[i]);
        }
        TreeUtils.setCubeDistances(this.tree, this.cube, cubeDistances);

        return 0.0;
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    @Override
    public void refresh() {
        this.cube = TreeUtils.getRandomCompatibleCube(this.tree.getRoot(), this.random);
    }

    public LinkedList<Integer> getCube() {
        return this.cube;
    }

}
