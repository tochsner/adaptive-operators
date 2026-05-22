package adapters;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.List;
import java.util.Random;

public class TaxaDistanceAdapter implements Adapter {

    private Tree tree;
    private int[] taxa;
    private Random random;

    public TaxaDistanceAdapter(Tree tree, int[] taxa) {
        this.tree = tree;
        this.taxa = taxa;
        this.random = new Random();
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
        return new double[] {
                Math.log(getDistance(
                        this.tree.getNode(this.taxa[0]),
                        this.tree.getNode(this.taxa[1])
                ))
        };
    }

    @Override
    public void update(double[] mutable, int nodeId) {
        TreeUtils.changeNodeDistance(
                this.tree.getNode(this.taxa[0]),
                this.tree.getNode(this.taxa[1]),
                Math.exp(mutable[0]),
                this.random
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

}
