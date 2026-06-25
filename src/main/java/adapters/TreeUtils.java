package adapters;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

import java.util.*;

public class TreeUtils {

    public static MRCA getCommonAncestor(Node nodeA, Node nodeB) {
        Set<Node> ancestorsA = new HashSet<>();
        Set<Node> path = new HashSet<>();

        Node current = nodeA;
        while (current != null) {
            ancestorsA.add(current);
            current = current.getParent();
        }

        Node mrca = nodeB;
        while (mrca != null && !ancestorsA.contains(mrca)) {
            path.add(mrca);
            mrca = mrca.getParent();
        }

        if (mrca == null) {
            throw new IllegalArgumentException("Nodes are not part of the same tree");
        }

        current = nodeA;
        while (current != mrca) {
            path.add(current);
            current = current.getParent();
        }

        return new MRCA(mrca, path);
    }

    public static double getDistance(Node nodeA, Node nodeB) {
        Node mrca = TreeUtils.getCommonAncestor(nodeA, nodeB).mrca();
        return 2.0 * mrca.getHeight() - nodeA.getHeight() - nodeB.getHeight();
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

    public static List<Integer> getAffectedDistances(Node nodeA, Node nodeB, LinkedList<Integer> cube) {
        Node root = nodeA.getTree().getRoot();
        Map<Node, Integer> mrcaMap = new HashMap<>();
        TreeUtils.getMRCAMap(root, cube, mrcaMap);

        MRCA mrca = TreeUtils.getCommonAncestor(nodeA, nodeB);

        List<Node> affectedNodes = new ArrayList<>();

        // we add all nodes on the way to the root

        Node curr = mrca.mrca;
        while (!curr.isRoot()) {
            curr = curr.getParent();
            affectedNodes.add(curr);
        }

        // we add all nodes on the path between node a and b

        affectedNodes.addAll(mrca.path);

        // get node IDs sorted by distance to MRCA

        affectedNodes = affectedNodes.stream()
                .sorted(Comparator.comparing(x -> Math.abs(x.getHeight() - mrca.mrca.getHeight())))
                .filter(x -> !x.isLeaf())
                .toList();

        // get cube indices

        List<Integer> affectedDistances = affectedNodes.stream().map(mrcaMap::get).toList();

        return affectedDistances;
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
            cube.addAll(rightCube);
            cube.addAll(leftCube);
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

    public static Set<Integer> getMRCAMap(Node root, LinkedList<Integer> cube, Map<Node, Integer> mrcaMap) {
        if (root.isLeaf()) {
            Set<Integer> nodes = new HashSet<>();
            nodes.add(root.getNr());
            return nodes;
        }

        Set<Integer> leftNodes = getMRCAMap(root.getLeft(), cube, mrcaMap);
        Set<Integer> rightNodes = getMRCAMap(root.getRight(), cube, mrcaMap);

        Set<Integer> allNodes = new HashSet<>();
        allNodes.addAll(leftNodes);
        allNodes.addAll(rightNodes);

        boolean seenLeft = false;
        boolean seenRight = false;

        for (Integer nodeId : cube) {
            if (leftNodes.contains(nodeId)) {
                if (seenRight && rightNodes.isEmpty()) {
                    mrcaMap.put(root, nodeId);
                    break ;
                } else if (seenRight) {
                    throw new RuntimeException("Incompatible cube detected.");
                } else {
                    seenLeft = true;
                    leftNodes.remove(nodeId);
                }
            }

            if (rightNodes.contains(nodeId)) {
                if (seenLeft && leftNodes.isEmpty()) {
                    mrcaMap.put(root, nodeId);
                    break;
                } else if (seenLeft) {
                    throw new RuntimeException("Incompatible cube detected.");
                } else {
                    seenRight = true;
                    rightNodes.remove(nodeId);
                }
            }
        }

        if (!mrcaMap.containsKey(root)) {
            throw new RuntimeException("Incompatible cube detected.");
        }

        return allNodes;
    }

    public static void reattachNode(Node node, double distanceTo1, Node reference1, Node reference2) {
        if (node.isRoot()) {
            throw new IllegalArgumentException("Cannot reattach the root node");
        }

        if (TreeUtils.containsNode(node, reference1) || TreeUtils.containsNode(node, reference2)) {
            throw new IllegalArgumentException("Reference nodes must not be inside the subtree being reattached");
        }

        MRCA initialMrca = TreeUtils.getCommonAncestor(reference1, reference2);
        double initialDistanceToMrcaFrom1 = initialMrca.mrca().getHeight() - reference1.getHeight();
        double initialDistanceToMrcaFrom2 = initialMrca.mrca().getHeight() - reference2.getHeight();
        double initialReferenceDistance = initialDistanceToMrcaFrom1 + initialDistanceToMrcaFrom2;

        if (distanceTo1 >= initialReferenceDistance) {
            throw new IllegalArgumentException("distanceTo1 must be smaller than the distance between the references");
        }

        double initialAttachmentHeight = getAttachmentHeight(
                reference1,
                reference2,
                initialMrca.mrca(),
                initialReferenceDistance,
                distanceTo1
        );

        if (initialAttachmentHeight < node.getHeight()) {
            throw new IllegalArgumentException("Cannot attach node below its current height");
        }

        Node attachmentParent = node.getParent();
        Node sibling = TreeUtils.getOtherChild(attachmentParent, node);
        Node oldGrandParent = attachmentParent.getParent();

        if (oldGrandParent == null) {
            sibling.setParent(null);
            if (node.getTree() != null) {
                node.getTree().setRootOnly(sibling);
            }
        } else {
            TreeUtils.replace(oldGrandParent, attachmentParent, sibling);
        }

        attachmentParent.removeChild(node);
        attachmentParent.removeChild(sibling);
        attachmentParent.setParent(null);
        node.setParent(null);

        MRCA mrca = TreeUtils.getCommonAncestor(reference1, reference2);
        double distanceToMrcaFrom1 = mrca.mrca().getHeight() - reference1.getHeight();
        double distanceToMrcaFrom2 = mrca.mrca().getHeight() - reference2.getHeight();
        double referenceDistance = distanceToMrcaFrom1 + distanceToMrcaFrom2;

        AttachmentPoint attachmentPoint = getAttachmentPoint(
                reference1,
                reference2,
                mrca.mrca(),
                referenceDistance,
                distanceTo1
        );

        attachmentParent.setHeight(attachmentPoint.height());

        if (attachmentPoint.parent() == null) {
            attachmentParent.addChild(node);
            attachmentParent.addChild(attachmentPoint.child());
            attachmentParent.setParent(null);
            if (node.getTree() != null) {
                node.getTree().setRootOnly(attachmentParent);
            }
        } else {
            TreeUtils.replace(attachmentPoint.parent(), attachmentPoint.child(), attachmentParent);
            attachmentParent.addChild(node);
            attachmentParent.addChild(attachmentPoint.child());
        }

        TreeUtils.markFilthy(node);
        TreeUtils.markFilthy(sibling);
        TreeUtils.markFilthy(attachmentParent);
        TreeUtils.markFilthy(attachmentPoint.child());
        TreeUtils.markFilthyToRoot(oldGrandParent == null ? sibling : oldGrandParent);
        TreeUtils.markFilthyToRoot(attachmentPoint.parent() == null ? attachmentParent : attachmentPoint.parent());
    }

    private record AttachmentPoint(Node child, Node parent, double height) {};

    private static AttachmentPoint getAttachmentPoint(
            Node reference1,
            Node reference2,
            Node mrca,
            double referenceDistance,
            double distanceTo1
    ) {
        double distanceToMrcaFrom1 = mrca.getHeight() - reference1.getHeight();

        if (distanceTo1 <= distanceToMrcaFrom1) {
            return getAttachmentPointAbove(reference1, getAttachmentHeight(
                    reference1,
                    reference2,
                    mrca,
                    referenceDistance,
                    distanceTo1
            ));
        }

        return getAttachmentPointAbove(reference2, getAttachmentHeight(
                reference1,
                reference2,
                mrca,
                referenceDistance,
                distanceTo1
        ));
    }

    private static double getAttachmentHeight(
            Node reference1,
            Node reference2,
            Node mrca,
            double referenceDistance,
            double distanceTo1
    ) {
        double distanceToMrcaFrom1 = mrca.getHeight() - reference1.getHeight();

        if (distanceTo1 <= distanceToMrcaFrom1) {
            return reference1.getHeight() + distanceTo1;
        }

        double distanceTo2 = referenceDistance - distanceTo1;
        return reference2.getHeight() + distanceTo2;
    }

    private static AttachmentPoint getAttachmentPointAbove(Node node, double targetHeight) {
        Node child = node;
        Node parent = child.getParent();

        while (parent != null && parent.getHeight() < targetHeight) {
            child = parent;
            parent = child.getParent();
        }

        if (child.getHeight() > targetHeight) {
            throw new IllegalArgumentException("Attachment height is below the target branch");
        }

        return new AttachmentPoint(child, parent, targetHeight);
    }

    public static boolean containsNode(Node root, Node query) {
        if (root == query) {
            return true;
        }

        for (Node child : root.getChildren()) {
            if (TreeUtils.containsNode(child, query)) {
                return true;
            }
        }

        return false;
    }

    private static void markFilthy(Node node) {
        if (node != null) {
            node.makeDirty(Tree.IS_FILTHY);
        }
    }

    private static void markFilthyToRoot(Node node) {
        while (node != null) {
            node.makeDirty(Tree.IS_FILTHY);
            node = node.getParent();
        }
    }

}
