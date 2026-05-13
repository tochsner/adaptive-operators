package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.List;

public class TreeTripletAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Integer> numberOfTripletsInput = new Input<>("numberOfTriplets",
            "number of taxon triplets to sample", 15);
    public final Input<Boolean> normalizeInput = new Input<>("normalize",
            "whether to divide triplet features by the current tree height", true);

    private Tree tree;
    private int[][] triplets;
    private boolean normalize;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.normalize = this.normalizeInput.get();
        this.triplets = sampleTriplets(this.tree.getLeafNodeCount(), this.numberOfTripletsInput.get());
    }

    @Override
    public int getNumImmutable() {
        return this.triplets.length;
    }

    @Override
    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        double[] immutable = new double[this.getNumImmutable()];
        double scale = getScale();

        for (int i = 0; i < this.triplets.length; i++) {
            int[] triplet = this.triplets[i];
            immutable[i] = getTripletFeature(triplet[0], triplet[1], triplet[2]) / scale;
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
            throw new IllegalArgumentException("TreeTripletAdapter has no mutable values");
        }
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0;
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
            throw new IllegalStateException("Cannot normalize triplet features for a tree with non-positive height");
        }

        return treeHeight;
    }

    private double getTripletFeature(int taxonA, int taxonB, int taxonC) {
        double lcaAB = getCommonAncestorHeight(taxonA, taxonB);
        double lcaAC = getCommonAncestorHeight(taxonA, taxonC);
        double lcaBC = getCommonAncestorHeight(taxonB, taxonC);

        return Math.min(lcaAC, lcaBC) - lcaAB;
    }

    private double getCommonAncestorHeight(int taxonA, int taxonB) {
        return getCommonAncestor(this.tree.getNode(taxonA), this.tree.getNode(taxonB)).getHeight();
    }

    private static Node getCommonAncestor(Node nodeA, Node nodeB) {
        while (nodeA != nodeB) {
            if (nodeB == null) {
                int a = 4;
            }

            if (nodeA.getHeight() < nodeB.getHeight()) {
                nodeA = nodeA.getParent();
            } else {
                nodeB = nodeB.getParent();
            }
        }

        return nodeA;
    }

    private static int[][] sampleTriplets(int taxonCount, int numberOfTriplets) {
        if (taxonCount < 3) {
            throw new IllegalArgumentException("TreeTripletAdapter requires at least three taxa");
        }

        if (numberOfTriplets < 0) {
            throw new IllegalArgumentException("numberOfTriplets must be non-negative");
        }

        int[][] triplets = new int[numberOfTriplets][3];

        for (int i = 0; i < numberOfTriplets; i++) {
            int[] shuffledTaxa = Randomizer.shuffled(taxonCount);
            triplets[i][0] = shuffledTaxa[0];
            triplets[i][1] = shuffledTaxa[1];
            triplets[i][2] = shuffledTaxa[2];
        }

        return triplets;
    }

}
