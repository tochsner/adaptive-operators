package adapters;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class TaxaDistanceAdapter implements Adapter {

    private Tree tree;
    private int[] taxa;
    public LinkedList<Integer> cube;
    private Random random;
    private double offset;

    public TaxaDistanceAdapter(Tree tree, int[] taxa) {
        this.tree = tree;
        this.taxa = taxa;
        this.random = new Random();
        this.offset = computeOffset();
    }

    private double computeOffset() {
        return Math.abs(
                this.tree.getNode(this.taxa[0]).getHeight()
                        - this.tree.getNode(this.taxa[1]).getHeight());
    }

    @Override
    public int getNumImmutable() {
        return 0;
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
        double distance = getDistance(
                this.tree.getNode(this.taxa[0]),
                this.tree.getNode(this.taxa[1]));
        return new double[] { Math.log(distance - this.offset) };
    }

    @Override
    public void update(double[] mutable, int nodeId) {
        TreeUtils.deterministicallyChangeNodeDistance(
                this.tree.getNode(this.taxa[0]),
                this.tree.getNode(this.taxa[1]),
                Math.exp(mutable[0]) + this.offset,
                this.cube
        );
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    private double getDistance(Node nodeA, Node nodeB) {
        Node mrca = TreeUtils.getCommonAncestor(nodeA, nodeB).mrca();
        return 2.0 * mrca.getHeight() - nodeA.getHeight() - nodeB.getHeight();
    }

    @Override
    public void refresh() {
        this.cube = TreeUtils.getRandomCompatibleCube(this.tree.getRoot(), this.taxa, this.random);
    }
}
