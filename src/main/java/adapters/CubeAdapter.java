package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.spec.evolution.tree.ClusterTree;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.RealVectorParam;
import mala.SubsetJukesCantorDistance;

import java.util.*;

public class CubeAdapter extends BEASTObject implements MAPAdapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");

    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    public final Input<RealScalarParam<?>> clockRateInput = new Input<>("clockRate", "", null, Input.Validate.OPTIONAL);
    public final Input<RealVectorParam<?>> clockRatesInput = new Input<>("clockRates", "", null, Input.Validate.OPTIONAL);

    private Tree tree;
    private RealScalarParam<?> clockRate;
    private RealVectorParam<?> clockRates;

    private LinkedList<Integer> cube;
    private Random random;

    private List<Tree> mapTrees;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.clockRate = this.clockRateInput.get();
        this.clockRates = this.clockRatesInput.get();
        this.random = new Random();

        this.initMapTrees();
        this.refresh();
    }

    private void initMapTrees() {
        this.mapTrees = new ArrayList<>();
        this.addMapTree(ClusterTree.Type.upgma);
    }

    private void addMapTree(ClusterTree.Type type) {
        for (int i = 0; i < 10; i++) {
            ClusterTree tree = new ClusterTree();
            tree.clusterTypeInput.setTypedValue(type, tree);
            tree.distanceInput.setTypedValue(new SubsetJukesCantorDistance(0.5), tree);
            tree.dataInput.setTypedValue(this.alignmentInput.get(), tree);
            tree.m_traitList.setTypedValue(this.tree.m_traitList.get(), tree);
            tree.initAndValidate();

            this.mapTrees.add(tree);
        }
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
        double logCorrection = 0.0;

        double[] cube = this.getMutable(nodeId);
        for (double v : cube) {
            logCorrection -= v;
        }

        return logCorrection;
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

    @Override
    public double[] getMutableMAP() {
        Tree randomMapTree = this.mapTrees.get(this.random.nextInt(this.mapTrees.size()));

        double[] cube = TreeUtils.getCubeDistances(this.tree, this.cube);
        double[] mapCube = TreeUtils.getCubeDistances(randomMapTree, this.cube);

        double sumDistances = Arrays.stream(cube).sum();
        double mapSumDistances = Arrays.stream(mapCube).sum();

        for (int i = 0; i < this.getNumMutable(); i++) {
            mapCube[i] *= sumDistances / mapSumDistances;
            mapCube[i] = Math.log(mapCube[i]);
        }

        return mapCube;
    }

}
