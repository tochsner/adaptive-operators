package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.*;

public class LocalTreeAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private Tree tree;

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
        Node node = this.tree.getNode(nodeId);
        if (node.isRoot()) throw new RuntimeException("Root encountered in LocalTreeAdapter.");

        Node parent = node.getParent();

        double branchToParent = parent == null ? 0.0 : parent.getHeight() - node.getHeight();
        double timeToRoot = this.tree.getRoot().getHeight() - node.getHeight();

        return new double[]{
                Math.log(branchToParent + 0.001),
                Math.log(timeToRoot + 0.001),
                Math.log(this.tree.getRoot().getHeight()),
        };
    }

    @Override
    public double[] getMutable(int nodeId) {
        Node node = this.tree.getNode(nodeId);
        if (node.isLeaf()) throw new RuntimeException("Leaf encountered in LocalTreeAdapter.");

        double shorterBranch = 0.0;

        if (node.getChildren().size() == 2) {
            double olderChildHeight = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
            shorterBranch = node.getHeight() - olderChildHeight;
        }

        if (shorterBranch < 0) {
            throw new RuntimeException("Negative branch encountered in LocalTreeAdapter.");
        }

        return new double[]{Math.log(shorterBranch)};
    }

    @Override
    public void update(double[] mutable, int nodeId) {
        Node node = this.tree.getNode(nodeId);
        if (node.isLeaf()) return;

        Map<Node, Double> newHeights = new HashMap<>();

        double newShorterBranch = Math.exp(mutable[0]);
        double olderChildHeight = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
        newHeights.put(node, olderChildHeight + newShorterBranch);

        // look at all nodes above. keep the shorter branch length the same

        while (node.getParent() != null) {
            Node parent = node.getParent();

            double previousOlderChildHeight = Math.max(parent.getLeft().getHeight(), parent.getRight().getHeight());
            double previousShorterBranch = parent.getHeight() - previousOlderChildHeight;

            double newOlderChildHeight = Math.max(
                    newHeights.getOrDefault(parent.getLeft(), parent.getLeft().getHeight()),
                    newHeights.getOrDefault(parent.getRight(), parent.getRight().getHeight())
            );
            double newParentHeight = newOlderChildHeight + previousShorterBranch;
            newHeights.put(parent, newParentHeight);

            node = parent;
        }

        // apply

        for (Node nodeToChange : newHeights.keySet()) {
            nodeToChange.setHeight(newHeights.get(nodeToChange));
        }

        for (Node nodeToChange : newHeights.keySet()) {
            if (nodeToChange.getParent() != null) {
                if (nodeToChange.getParent().getHeight() < nodeToChange.getHeight())
                    throw new RuntimeException("A");
            }

            for (Node c : nodeToChange.getChildren()) {
                if (nodeToChange.getHeight() < c.getHeight())
                    throw new RuntimeException("B");
            }
        }
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return -this.getMutable(nodeId)[0];
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
