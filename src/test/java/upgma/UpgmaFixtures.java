package upgma;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.type.RealScalar;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Trees and data matrices shared by the {@code upgma} tests.
 *
 * <p>The trees are built by drawing a random clustering and writing it into a tree, which is the
 * same shape of input {@link UpgmaSplitModel} sees in the operator: strictly increasing merge
 * heights and no ties.
 */
final class UpgmaFixtures {

    private UpgmaFixtures() {
    }

    static RealScalar<?> rate(double value) {
        return new RealScalarParam<>(value, PositiveReal.INSTANCE);
    }

    /** A random ultrametric tree on {@code leafCount} contemporaneous tips. */
    static Tree randomTree(int leafCount, Random random) {
        return parse(randomNewick(leafCount, random));
    }

    /** Symmetric matrix with a zero diagonal, standing in for the JC69 distances. */
    static double[][] randomSymmetric(int leafCount, Random random) {
        double[][] matrix = new double[leafCount][leafCount];
        for (int i = 0; i < leafCount; i++) {
            for (int j = i + 1; j < leafCount; j++) {
                double value = 0.05 + random.nextDouble();
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
        return matrix;
    }

    static Tree parse(String newick) {
        TreeParser tree = new TreeParser();
        tree.initByName("IsLabelledNewick", true, "newick", newick, "adjustTipHeights", false);
        return tree;
    }

    /**
     * Joins two random clusters at a time, at strictly increasing heights, and writes the result out
     * as Newick. Leaf {@code i} is labelled {@code Ti}, and {@code TreeParser} numbers the leaves in
     * label order, so leaf node number {@code i} is cluster {@code i} throughout the tests.
     */
    private static String randomNewick(int leafCount, Random random) {
        List<String> subtree = new ArrayList<>();
        List<Double> subtreeHeight = new ArrayList<>();
        for (int leaf = 0; leaf < leafCount; leaf++) {
            subtree.add("T" + leafLabel(leaf, leafCount));
            subtreeHeight.add(0.0);
        }

        double height = 0.0;
        while (subtree.size() > 1) {
            height += 0.1 + random.nextDouble();

            int left = random.nextInt(subtree.size());
            int right = (left + 1 + random.nextInt(subtree.size() - 1)) % subtree.size();
            int first = Math.min(left, right);
            int second = Math.max(left, right);

            String joined = "(" + subtree.get(first) + ":" + (height - subtreeHeight.get(first))
                    + "," + subtree.get(second) + ":" + (height - subtreeHeight.get(second)) + ")";

            subtree.set(first, joined);
            subtreeHeight.set(first, height);
            subtree.remove(second);
            subtreeHeight.remove(second);
        }

        return subtree.get(0) + ":0.0;";
    }

    /** Zero-pads the label so that lexicographic order matches numeric order. */
    private static String leafLabel(int leaf, int leafCount) {
        int width = String.valueOf(leafCount - 1).length();
        return String.format("%0" + width + "d", leaf);
    }

    /** Every clade of the tree, as a sorted leaf list, paired with the height of its node. */
    static List<String> cladesWithHeights(Tree tree) {
        List<String> clades = new ArrayList<>();
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                List<Integer> leaves = new ArrayList<>();
                for (Node leaf : node.getAllLeafNodes()) {
                    leaves.add(leaf.getNr());
                }
                leaves.sort(null);
                clades.add(leaves + "@" + String.format("%.9f", node.getHeight()));
            }
        }
        clades.sort(null);
        return clades;
    }

    /** The same description, built from a clustering result rather than a tree. */
    static List<String> cladesWithHeights(UpgmaClustering.Result result, int leafCount) {
        List<List<Integer>> members = new ArrayList<>();
        for (int leaf = 0; leaf < leafCount; leaf++) {
            members.add(new ArrayList<>(List.of(leaf)));
        }

        List<String> clades = new ArrayList<>();
        for (int merge = 0; merge < leafCount - 1; merge++) {
            List<Integer> joined = new ArrayList<>(members.get(result.mergeLeft()[merge]));
            joined.addAll(members.get(result.mergeRight()[merge]));
            joined.sort(null);
            members.add(joined);
            clades.add(joined + "@" + String.format("%.9f", result.mergeHeight()[merge]));
        }
        clades.sort(null);
        return clades;
    }

    /** Leaf numbers under every node, indexed by node number. */
    static int[][] clades(Tree tree) {
        int[][] clades = new int[tree.getNodeCount()][];
        for (Node node : tree.getNodesAsArray()) {
            if (node.isLeaf()) {
                clades[node.getNr()] = new int[]{node.getNr()};
            }
        }
        collectClades(tree.getRoot(), clades);
        return clades;
    }

    private static int[] collectClades(Node node, int[][] clades) {
        if (node.isLeaf()) {
            return clades[node.getNr()];
        }

        int[] left = collectClades(node.getChild(0), clades);
        int[] right = collectClades(node.getChild(1), clades);
        int[] leaves = new int[left.length + right.length];
        System.arraycopy(left, 0, leaves, 0, left.length);
        System.arraycopy(right, 0, leaves, left.length, right.length);
        clades[node.getNr()] = leaves;
        return leaves;
    }
}
