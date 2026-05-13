package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.List;

public class NodePositionAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Integer> numberOfTaxaInput = new Input<>("numberOfTaxa",
            "number of taxa to sample for node distance features", 15);
    public final Input<Boolean> normalizeInput = new Input<>("normalize",
            "whether to divide distances by twice the current tree height", true);

    private Tree tree;
    private int[] taxa;
    private boolean normalize;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.normalize = this.normalizeInput.get();
        this.taxa = sampleTaxa(this.tree.getLeafNodeCount(), this.numberOfTaxaInput.get());
    }

    @Override
    public int getNumImmutable() {
        return this.taxa.length;
    }

    @Override
    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        Node node = this.tree.getNode(nodeId);
        double[] immutable = new double[this.getNumImmutable()];
        double scale = getScale();

        for (int i = 0; i < this.taxa.length; i++) {
            immutable[i] = getDistance(node, this.tree.getNode(this.taxa[i])) / scale;
        }

        return immutable;
    }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[0];
    }

    @Override
    public void update(double[] mutable, int nodeId) {
        if (mutable.length != 0) {
            throw new IllegalArgumentException("NodePositionAdapter has no mutable values");
        }
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0.0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    private double getScale() {
        if (!this.normalize) {
            return 1.0;
        }

        double treeHeight = this.tree.getRoot().getHeight();

        if (treeHeight <= 0.0) {
            throw new IllegalStateException("Cannot normalize node distances for a tree with non-positive height");
        }

        return 2.0 * treeHeight;
    }

    private static double getDistance(Node nodeA, Node nodeB) {
        Node commonAncestor = getCommonAncestor(nodeA, nodeB);
        return 2.0 * commonAncestor.getHeight() - nodeA.getHeight() - nodeB.getHeight();
    }

    private static Node getCommonAncestor(Node nodeA, Node nodeB) {
        while (nodeA != nodeB) {
            if (nodeA.getHeight() < nodeB.getHeight()) {
                nodeA = nodeA.getParent();
            } else {
                nodeB = nodeB.getParent();
            }
        }

        return nodeA;
    }

    private static int[] sampleTaxa(int taxonCount, int numberOfTaxa) {
        if (taxonCount < 1) {
            throw new IllegalArgumentException("NodePositionAdapter requires at least one taxon");
        }

        if (numberOfTaxa < 0) {
            throw new IllegalArgumentException("numberOfTaxa must be non-negative");
        }

        int[] shuffledTaxa = Randomizer.shuffled(taxonCount);
        int[] sampledTaxa = new int[Math.min(numberOfTaxa, taxonCount)];
        System.arraycopy(shuffledTaxa, 0, sampledTaxa, 0, sampledTaxa.length);

        return sampledTaxa;
    }

}
