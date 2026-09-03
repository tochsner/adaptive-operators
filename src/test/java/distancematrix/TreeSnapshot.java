package distancematrix;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

/**
 * Stands in for what BEAST's {@code State} does for a real chain: remembers the tree so a rejected
 * proposal can be undone. BEAST calls {@code operator.reject} before {@code state.restore}, so tests
 * that reject a proposal have to restore the tree themselves or the tree and the operator's matrix
 * drift apart.
 */
class TreeSnapshot {

    private final int[] leftChild;
    private final int[] rightChild;
    private final double[] height;
    private int rootNr;

    TreeSnapshot(Tree tree) {
        this.leftChild = new int[tree.getNodeCount()];
        this.rightChild = new int[tree.getNodeCount()];
        this.height = new double[tree.getNodeCount()];
    }

    void store(Tree tree) {
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            this.height[nodeNr] = node.getHeight();

            if (node.isLeaf()) {
                this.leftChild[nodeNr] = -1;
                this.rightChild[nodeNr] = -1;
            } else {
                this.leftChild[nodeNr] = node.getChild(0).getNr();
                this.rightChild[nodeNr] = node.getChild(1).getNr();
            }
        }

        this.rootNr = tree.getRoot().getNr();
    }

    void restore(Tree tree) {
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            node.setHeight(this.height[nodeNr]);

            if (this.leftChild[nodeNr] < 0) {
                continue;
            }

            node.removeAllChildren(false);
            node.addChild(tree.getNode(this.leftChild[nodeNr]));
            node.addChild(tree.getNode(this.rightChild[nodeNr]));
        }

        Node root = tree.getNode(this.rootNr);
        root.setParent(null);
        tree.setRootOnly(root);
    }
}
