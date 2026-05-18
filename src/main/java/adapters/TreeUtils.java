package adapters;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class TreeUtils {

    public static MRCA getCommonAncestor(Node nodeA, Node nodeB) {
        Set<Node> path = new HashSet<>();

        while (nodeA != nodeB) {
            path.add(nodeA);
            path.add(nodeB);

            if (nodeA.getHeight() < nodeB.getHeight()) {
                nodeA = nodeA.getParent();
            } else {
                nodeB = nodeB.getParent();
            }
        }

        return new MRCA(nodeA, path);
    }

    public record MRCA(Node mrca, Set<Node> path) {};

    /**
     * @param parent the parent
     * @param child  the child that you want the sister of
     * @return the other child of the given parent.
     */
    public static Node getOtherChild(final Node parent, final Node child) {
        if (parent.getLeft().getNr() == child.getNr()) {
            return parent.getRight();
        } else {
            return parent.getLeft();
        }
    }

    /**
     * replace child with another node
     *
     * @param node
     * @param child
     * @param replacement
     */
    public static void replace(final Node node, final Node child, final Node replacement) {
        node.removeChild(child);
        node.addChild(replacement);
        node.makeDirty(Tree.IS_FILTHY);
        replacement.makeDirty(Tree.IS_FILTHY);
    }

    /**
     * Changes the tree such that the distance of the two given taxa {@code nodeA} and {@code nodeB} is equal
     * to {@code newDistance}.
     * This is non-deterministic. It is equivalent to choosing a taxa ordering with (nodeA, nodeB) next to each other
     * uniformly at random and then changing the (nodeA, nodeB) distance while preserving the order.
     * Note that this only works for ultrametric trees.
     */
    public static double changeNodeDistance(Node nodeA, Node nodeB, double newDistance, Random random) {
        MRCA mrcaInfo = TreeUtils.getCommonAncestor(nodeA, nodeB);
        Node mrca = mrcaInfo.mrca();
        Set<Node> pathBetweenAB = mrcaInfo.path();

        double currentDistance = 2.0 * mrca.getHeight() - nodeA.getHeight() - nodeB.getHeight();
        double newMrcaHeight = (newDistance + nodeA.getHeight() + nodeB.getHeight()) / 2.0;

        if (currentDistance < newDistance) {
            // we increase the distance

            int dof = 0;

            while (true) {
                Node mrcaParent = mrca.getParent();

                if (mrcaParent == null) {
                    // the mrca is already the root
                    // we simply increase its height
                    mrca.setHeight(newMrcaHeight);
                    return dof * Math.log(0.5);
                }

                if (newMrcaHeight < mrcaParent.getHeight()) {
                    // the parent of the MRCA is older, we simply increase the height of the MRCA
                    mrca.setHeight(newMrcaHeight);
                    return dof * Math.log(0.5);
                }

                // the parent of the MRCA is younger

                // we switch and continue looking

                // we randomly choose a subtree to pair with the other parent subtree

                Node chosenSubtree = random.nextBoolean() ? mrca.getLeft() : mrca.getRight();
                dof++;

                Node nonChosenSubtree = TreeUtils.getOtherChild(mrca, chosenSubtree);
                Node otherParentSubtree = TreeUtils.getOtherChild(mrcaParent, mrca);

                // we keep the order of mrca and mrcaParent (mrca is younger than mrcaParent)
                // we thus swap them

                mrca.setHeight(mrcaParent.getHeight());

                TreeUtils.replace(mrca, nonChosenSubtree, otherParentSubtree);
                TreeUtils.replace(mrcaParent, otherParentSubtree, nonChosenSubtree);

                // start from the beginning with the new MRCA
                mrca = mrcaParent;
            }
        } else {
            // we decrease the distance

            int dof = 0;

            while (true) {
                // the obstacle node is the older child

                Node obstacle;
                if (mrca.getLeft().getHeight() < mrca.getRight().getHeight()) {
                    obstacle = mrca.getRight();
                } else {
                    obstacle = mrca.getLeft();
                }

                if (obstacle.getHeight() < newMrcaHeight) {
                    // neither of the children is younger than the new mrca height
                    // we simply move the mrca
                    mrca.setHeight(newMrcaHeight);
                    return -dof * Math.log(0.5);
                }

                // assert that obstacle is an internal node (this should be the case for ultrametric trees)

                if (obstacle.isLeaf()) {
                    throw new RuntimeException("Obstacle leaf detected. Is the tree not ultrametric?");
                }

                // we walk along the subtree which contains either nodeA or nodeB

                Node subtreeToTraverse = pathBetweenAB.contains(obstacle.getLeft()) ? obstacle.getLeft() : obstacle.getRight();
                Node subtreeNotToTraverse = TreeUtils.getOtherChild(obstacle, subtreeToTraverse);
                Node nonObstacleSubtree = TreeUtils.getOtherChild(mrca, obstacle);

                dof++;

                // we keep the order of mrca and obstacle (mrca is older) by swapping

                mrca.setHeight(obstacle.getHeight());

                TreeUtils.replace(mrca, nonObstacleSubtree, subtreeNotToTraverse);
                TreeUtils.replace(obstacle, subtreeNotToTraverse, nonObstacleSubtree);

                // start from the beginning with the new MRCA
                mrca = obstacle;
            }
        }
    }

}
