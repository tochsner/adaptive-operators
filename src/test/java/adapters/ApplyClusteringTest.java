package adapters;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ApplyClusteringTest {

    @Test
    void writingBackTheSameTopologyChangesNothing() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):2.0,C:3.0):0.0;");
        Map<Node, BitSet> before = clades(tree.getRoot());

        // clusters 0..2 are the leaves, 3 joins the first two, 4 joins that with the third
        TreeUtils.applyClustering(
                tree,
                new int[]{leaf(tree, "A"), 3},
                new int[]{leaf(tree, "B"), leaf(tree, "C")},
                new double[]{1.0, 3.0});

        assertThat(tree.getRoot().toNewick()).isEqualTo("((A:1.0,B:1.0):2.0,C:3.0):0.0");
        assertThat(clades(tree.getRoot())).isEqualTo(before);
        assertTreeIsWellFormed(tree.getRoot(), "A", "B", "C");
    }

    @Test
    void changingTheTopologyKeepsTheNodesOfSurvivingClades() {
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):2.0,D:4.0):0.0;");

        Node cladeAB = findLeaf(tree.getRoot(), "A").getParent();
        Node root = tree.getRoot();

        // regroup C with D, leaving the AB clade and the root clade intact
        TreeUtils.applyClustering(
                tree,
                new int[]{leaf(tree, "A"), leaf(tree, "C"), 4},
                new int[]{leaf(tree, "B"), leaf(tree, "D"), 5},
                new double[]{1.0, 2.0, 4.0});

        assertThat(tree.getRoot().toNewick()).isEqualTo("((A:1.0,B:1.0):3.0,(C:2.0,D:2.0):2.0):0.0");
        assertThat(findLeaf(tree.getRoot(), "A").getParent())
                .as("the AB clade survived, so it keeps its node")
                .isSameAs(cladeAB);
        assertThat(tree.getRoot()).as("the root clade always survives").isSameAs(root);
        assertTreeIsWellFormed(tree.getRoot(), "A", "B", "C", "D");
    }

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L})
    void randomClusteringsProduceWellFormedTreesAndKeepSurvivingClades(long seed) {
        Random random = new Random(seed);
        TreeParser tree = parseTree("((((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):1.0,E:4.0):0.0;");
        String[] leafIds = {"A", "B", "C", "D", "E"};

        for (int step = 0; step < 200; step++) {
            Map<BitSet, Node> nodeOfCladeBefore = new HashMap<>();
            for (Map.Entry<Node, BitSet> entry : clades(tree.getRoot()).entrySet()) {
                nodeOfCladeBefore.put(entry.getValue(), entry.getKey());
            }

            int[] mergeLeft = new int[4];
            int[] mergeRight = new int[4];
            double[] mergeHeight = new double[4];
            randomClustering(5, random, mergeLeft, mergeRight, mergeHeight);

            TreeUtils.applyClustering(tree, mergeLeft, mergeRight, mergeHeight);

            assertTreeIsWellFormed(tree.getRoot(), leafIds);

            for (Map.Entry<Node, BitSet> entry : clades(tree.getRoot()).entrySet()) {
                Node nodeBefore = nodeOfCladeBefore.get(entry.getValue());

                if (nodeBefore != null) {
                    assertThat(entry.getKey())
                            .as("clade " + entry.getValue() + " survived step " + step)
                            .isSameAs(nodeBefore);
                }
            }
        }
    }

    @Test
    void rejectsTheWrongNumberOfMerges() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):2.0,C:3.0):0.0;");

        assertThatThrownBy(() -> TreeUtils.applyClustering(
                tree, new int[]{0}, new int[]{1}, new double[]{1.0}))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("expected 2 merges");
    }

    /** Draws a random clustering: repeatedly join two of the remaining clusters, going upwards. */
    private static void randomClustering(
            int leafCount, Random random, int[] mergeLeft, int[] mergeRight, double[] mergeHeight) {
        List<Integer> open = new ArrayList<>();
        for (int leaf = 0; leaf < leafCount; leaf++) {
            open.add(leaf);
        }

        double height = 0.0;
        for (int merge = 0; merge < leafCount - 1; merge++) {
            int left = random.nextInt(open.size());
            int right = (left + 1 + random.nextInt(open.size() - 1)) % open.size();

            mergeLeft[merge] = open.get(left);
            mergeRight[merge] = open.get(right);
            height += 0.1 + random.nextDouble();
            mergeHeight[merge] = height;

            open.remove(Math.max(left, right));
            open.remove(Math.min(left, right));
            open.add(leafCount + merge);
        }
    }

    private static int leaf(TreeParser tree, String id) {
        return findLeaf(tree.getRoot(), id).getNr();
    }

    private static Map<Node, BitSet> clades(Node root) {
        Map<Node, BitSet> clades = new HashMap<>();
        collectClades(root, clades);
        return clades;
    }

    private static BitSet collectClades(Node node, Map<Node, BitSet> clades) {
        BitSet clade = new BitSet();

        if (node.isLeaf()) {
            clade.set(node.getNr());
        } else {
            for (Node child : node.getChildren()) {
                clade.or(collectClades(child, clades));
            }
        }

        clades.put(node, clade);
        return clade;
    }

    private static TreeParser parseTree(String newick) {
        TreeParser tree = new TreeParser();
        tree.initByName(
                "IsLabelledNewick", true,
                "newick", newick,
                "adjustTipHeights", false
        );
        return tree;
    }

    private static Node findLeaf(Node node, String id) {
        if (node.isLeaf() && id.equals(node.getID())) {
            return node;
        }
        for (Node child : node.getChildren()) {
            Node leaf = findLeaf(child, id);
            if (leaf != null) {
                return leaf;
            }
        }
        return null;
    }

    private static void assertTreeIsWellFormed(Node root, String... expectedLeafIds) {
        Set<Node> visited = new HashSet<>();
        Set<String> leafIds = new HashSet<>();
        collectNodes(root, null, visited, leafIds);

        assertThat(leafIds).containsExactlyInAnyOrder(expectedLeafIds);
        assertThat(visited).hasSize(2 * expectedLeafIds.length - 1);
    }

    private static void collectNodes(Node node, Node expectedParent, Set<Node> visited, Set<String> leafIds) {
        assertThat(node.getParent()).isSameAs(expectedParent);
        assertThat(visited.add(node)).isTrue();

        if (node.isLeaf()) {
            leafIds.add(node.getID());
        } else {
            assertThat(node.getChildren()).hasSize(2);
            for (Node child : node.getChildren()) {
                assertThat(child.getHeight())
                        .as("child sits below its parent")
                        .isLessThan(node.getHeight());
            }
        }

        for (Node child : node.getChildren()) {
            collectNodes(child, node, visited, leafIds);
        }
    }
}
