package adapters;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

import java.util.*;

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

    public static double[] getCubeDistances(Tree tree, LinkedList<Integer> cube) {
        double[] distances = new double[cube.size() - 1];

        for (int i = 0; i < cube.size() - 1; i++) {
            Node nodeA = tree.getNode(cube.get(i));
            Node nodeB = tree.getNode(cube.get(i + 1));
            Node mrca = TreeUtils.getCommonAncestor(nodeA, nodeB).mrca;
            distances[i] = 2.0 * mrca.getHeight() - nodeA.getHeight() - nodeB.getHeight();
        }

        return distances;
    }

    public static void setCubeDistances(Tree tree, LinkedList<Integer> cube, double[] cubeDistances) {
        for (int i = 0; i < cube.size() - 1; i++) {
            Node nodeA = tree.getNode(cube.get(i));
            Node nodeB = tree.getNode(cube.get(i + 1));
            TreeUtils.deterministicallyChangeNodeDistance(nodeA, nodeB, cubeDistances[i], cube);
        }
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

    /**
     * Changes the tree such that the distance of the two given taxa {@code nodeA} and {@code nodeB} is equal
     * to {@code newDistance}.
     * This is deterministic. The operation will preserve the given cube (node ordering). Note that the cube must
     * compatible with the tree.
     */
    public static double deterministicallyChangeNodeDistance(Node nodeA, Node nodeB, double newDistance, LinkedList<Integer> cube) {
        MRCA mrcaInfo = TreeUtils.getCommonAncestor(nodeA, nodeB);
        Node mrca = mrcaInfo.mrca();

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

                Node otherParentSubtree = TreeUtils.getOtherChild(mrcaParent, mrca);

                // we choose the subtree next to otherParentSubtree in the cube to pair with the other parent subtree

                Node chosenSubtree = TreeUtils.getNeighborSubTree(cube, otherParentSubtree, mrca.getLeft(), mrca.getRight());
                dof++;

                Node nonChosenSubtree = TreeUtils.getOtherChild(mrca, chosenSubtree);


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
            // this is always deterministic. we fall back to the normal version
            return TreeUtils.changeNodeDistance(nodeA, nodeB, newDistance, null);
        }
    }

    private static Node getNeighborSubTree(LinkedList<Integer> cube, Node query, Node candidateA, Node candidateB) {
        Set<Integer> queryNodes = TreeUtils.getAllNodes(query);
        Set<Integer> candidateANodes = TreeUtils.getAllNodes(candidateA);
        Set<Integer> candidateBNodes = TreeUtils.getAllNodes(candidateB);

        Node lastEncountered = null;
        for (Integer nodeId : cube) {
            if (queryNodes.contains(nodeId)) {

                if (lastEncountered == candidateA) {
                    return candidateA;
                } else if (lastEncountered == candidateB) {
                    return candidateB;
                } else {
                    lastEncountered = query;
                }

            } else if (candidateANodes.contains(nodeId)) {

                if (lastEncountered == query) {
                    return candidateA;
                } else {
                    lastEncountered = candidateA;
                }

            } else if (candidateBNodes.contains(nodeId)) {

                if (lastEncountered == query) {
                    return candidateB;
                } else {
                    lastEncountered = candidateB;
                }

            } else {

                lastEncountered = null;

            }

        }

        throw new RuntimeException("Neighboring subtree not found. This should not happen.");
    }

    private static Set<Integer> getAllNodes(Node root) {
        if (root.isLeaf()) {
            Set<Integer> nodes = new HashSet<>();
            nodes.add(root.getNr());
            return nodes;
        }

        Set<Integer> nodes = getAllNodes(root.getLeft());
        nodes.addAll(getAllNodes(root.getRight()));

        return nodes;
    }

    public static LinkedList<Integer> getRandomCompatibleCube(Node root, Random random) {
        if (root.isLeaf()) {
            LinkedList<Integer> cube = new LinkedList<>();
            cube.add(root.getNr());
            return cube;
        }

        LinkedList<Integer> cube;

        if (random.nextBoolean()) {
            cube = getRandomCompatibleCube(root.getLeft(), random);
            cube.addAll(getRandomCompatibleCube(root.getRight(), random));
        } else {
            cube = getRandomCompatibleCube(root.getRight(), random);
            cube.addAll(getRandomCompatibleCube(root.getLeft(), random));
        }

        return cube;
    }

    public static LinkedList<Integer> getRandomCompatibleCube(Node root, int[] taxa, Random random)  {
        LinkedList<Integer> cube = new LinkedList<>();
        TreeUtils.getRandomCompatibleCube(root, cube, taxa, random);
        return cube;
    }

    public static boolean getRandomCompatibleCube(Node root, LinkedList<Integer> cube, int[] taxa, Random random) {
        if (root.isLeaf()) {
            cube.add(root.getNr());
            return root.getNr() == taxa[0] || root.getNr() == taxa[1];
        }

        LinkedList<Integer> leftCube = new LinkedList<>();
        boolean leftHasTaxa = getRandomCompatibleCube(root.getLeft(), leftCube, taxa, random);

        LinkedList<Integer> rightCube = new LinkedList<>();
        boolean rightHasTaxa = getRandomCompatibleCube(root.getRight(), rightCube, taxa, random);

        if (leftHasTaxa && rightHasTaxa && random.nextBoolean()) {
            Collections.reverse(leftCube);
            cube.addAll(leftCube);
            cube.addAll(rightCube);
        } else if (leftHasTaxa && rightHasTaxa) {
            Collections.reverse(rightCube);
            cube.addAll(rightCube);
            cube.addAll(leftCube);
        } else if (leftHasTaxa) {
            cube.addAll(leftCube);
            cube.addAll(rightCube);
        } else if (rightHasTaxa) {
            cube.addAll(rightCube);
            cube.addAll(leftCube);
        } else if (random.nextBoolean()) {
            cube.addAll(leftCube);
            cube.addAll(rightCube);
        } else {
            cube.addAll(leftCube);
            cube.addAll(rightCube);
        }

        return leftHasTaxa || rightHasTaxa;
    }

    public static boolean isCompatible(Node root, LinkedList<Integer> cube) {
        if (root.isLeaf()) return true;

        Set<Integer> leftNodes = TreeUtils.getAllNodes(root.getLeft());
        Set<Integer> rightNodes = TreeUtils.getAllNodes(root.getRight());

        boolean seenLeft = false;
        boolean seenRight = false;

        for (Integer nodeId : cube) {
            if (leftNodes.contains(nodeId)) {
                if (seenRight && rightNodes.isEmpty()) {
                    break;
                } else if (seenRight) {
                    return false;
                } else {
                    seenLeft = true;
                    leftNodes.remove(nodeId);
                }
            }

            if (rightNodes.contains(nodeId)) {
                if (seenLeft && leftNodes.isEmpty()) {
                    break;
                } else if (seenLeft) {
                    return false;
                } else {
                    seenRight = true;
                    rightNodes.remove(nodeId);
                }
            }
        }

        return isCompatible(root.getLeft(), cube) && isCompatible(root.getRight(), cube);
    }

}
