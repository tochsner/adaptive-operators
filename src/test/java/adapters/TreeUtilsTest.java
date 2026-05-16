package adapters;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class TreeUtilsTest {

    @Test
    void changeNodeDistanceCanDecreaseDistanceWithoutChangingTopology() {
        TreeParser tree = parseTree("((A:3.0,B:3.0):1.0,C:4.0):0.0;");
        Node nodeA = findLeaf(tree.getRoot(), "A");
        Node nodeB = findLeaf(tree.getRoot(), "B");

        TreeUtils.changeNodeDistance(nodeA, nodeB, 2.0, getRandom());

        assertThat(distance(nodeA, nodeB)).isCloseTo(2.0, within(1e-12));
        assertThat(tree.getRoot().toNewick()).isEqualTo("((A:1.0,B:1.0):3.0,C:4.0):0.0");
        assertTreeIsConnected(tree.getRoot(), "A", "B", "C");
    }

    @Test
    void changeNodeDistanceCanIncreaseDistanceWithinParentHeightWithoutChangingTopology() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):3.0,C:4.0):0.0;");
        Node nodeA = findLeaf(tree.getRoot(), "A");
        Node nodeB = findLeaf(tree.getRoot(), "B");

        TreeUtils.changeNodeDistance(nodeA, nodeB, 6.0, getRandom());

        assertThat(distance(nodeA, nodeB)).isCloseTo(6.0, within(1e-12));
        assertThat(tree.getRoot().toNewick()).isEqualTo("((A:3.0,B:3.0):1.0,C:4.0):0.0");
        assertTreeIsConnected(tree.getRoot(), "A", "B", "C");
    }

    @Test
    void changeNodeDistanceCanIncreaseDistanceThroughParentRotation1() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
        Node nodeA = findLeaf(tree.getRoot(), "A");
        Node nodeB = findLeaf(tree.getRoot(), "B");

        TreeUtils.changeNodeDistance(nodeA, nodeB, 6.0, getRandom());

        assertThat(distance(nodeA, nodeB)).isCloseTo(6.0, within(1e-12));
        assertThat(tree.getRoot().toNewick()).isEqualTo("((A:2.0,C:2.0):1.0,B:3.0):0.0");
        assertTreeIsConnected(tree.getRoot(), "A", "B", "C");
    }

    @Test
    void changeNodeDistanceCanIncreaseDistanceThroughFourTaxonRotation2() {
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):3.0,D:5.0):0.0;");
        Node nodeA = findLeaf(tree.getRoot(), "A");
        Node nodeB = findLeaf(tree.getRoot(), "B");

        TreeUtils.changeNodeDistance(nodeA, nodeB, 6.0, getRandom());

        assertThat(distance(nodeA, nodeB)).isCloseTo(6.0, within(1e-12));
        assertThat(tree.getRoot().toNewick()).isEqualTo("(((A:2.0,C:2.0):1.0,B:3.0):2.0,D:5.0):0.0");
        assertTreeIsConnected(tree.getRoot(), "A", "B", "C", "D");
    }

    @Test
    void changeNodeDistanceCanDecreaseDistanceThroughFourTaxonRotation1() {
        TreeParser tree = parseTree("(((A:1.0,B:1.0):2.0,C:3.0):3.0,D:6.0):0.0;");
        Node nodeA = findLeaf(tree.getRoot(), "A");
        Node nodeD = findLeaf(tree.getRoot(), "D");

        TreeUtils.changeNodeDistance(nodeA, nodeD, 4.0, getRandom());

        assertThat(distance(nodeA, nodeD)).isCloseTo(4.0, within(1e-12));
        assertThat(tree.getRoot().toNewick()).isEqualTo("(((A:1.0,B:1.0):1.0,D:2.0):1.0,C:3.0):0.0");
        assertTreeIsConnected(tree.getRoot(), "A", "B", "C", "D");
    }

    @Test
    void changeNodeDistanceCanDecreaseDistanceThroughParentRotation2() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):2.0,(C:2.0,D:2.0):1.0):0.0;");
        Node nodeA = findLeaf(tree.getRoot(), "A");
        Node nodeD = findLeaf(tree.getRoot(), "D");

        TreeUtils.changeNodeDistance(nodeA, nodeD, 2.5, getRandom());

        assertThat(distance(nodeA, nodeD)).isCloseTo(2.5, within(1e-12));
        assertTreeIsConnected(tree.getRoot(), "A", "B", "C", "D");
    }

    @ParameterizedTest(name = "{0}, seed={1}")
    @MethodSource("randomOrderingCases")
    void changeNodeDistanceIsCorrectForDifferentRandomOrderings(
            String caseName,
            long seed,
            String newick,
            String firstTaxon,
            String secondTaxon,
            double targetDistance,
            String[] expectedLeafIds
    ) {
        TreeParser tree = parseTree(newick);
        Node firstNode = findLeaf(tree.getRoot(), firstTaxon);
        Node secondNode = findLeaf(tree.getRoot(), secondTaxon);

        TreeUtils.changeNodeDistance(firstNode, secondNode, targetDistance, new Random(seed));

        assertThat(distance(firstNode, secondNode)).isCloseTo(targetDistance, within(1e-12));
        assertTreeIsConnected(tree.getRoot(), expectedLeafIds);
    }

    private static Stream<Arguments> randomOrderingCases() {
        long[] seeds = {1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 127L, 128L};
        DistanceCase[] cases = {
                new DistanceCase(
                        "pectinate increase across two rotations",
                        "(((A:1.0,B:1.0):1.0,C:2.0):3.0,D:5.0):0.0;",
                        "A",
                        "B",
                        6.0,
                        "A",
                        "B",
                        "C",
                        "D"
                ),
                new DistanceCase(
                        "pectinate decrease across two rotations",
                        "(((A:1.0,B:1.0):2.0,C:3.0):3.0,D:6.0):0.0;",
                        "A",
                        "D",
                        3.5,
                        "A",
                        "B",
                        "C",
                        "D"
                ),
                new DistanceCase(
                        "balanced decrease through internal obstacle",
                        "((A:1.0,B:1.0):2.0,(C:2.0,D:2.0):1.0):0.0;",
                        "A",
                        "D",
                        2.5,
                        "A",
                        "B",
                        "C",
                        "D"
                ),
                new DistanceCase(
                        "opposite pectinate increase",
                        "(A:5.0,(B:2.0,(C:1.0,D:1.0):1.0):3.0):0.0;",
                        "C",
                        "D",
                        6.0,
                        "A",
                        "B",
                        "C",
                        "D"
                ),
                new DistanceCase(
                        "balanced increase from cherry",
                        "((A:1.0,B:1.0):4.0,(C:2.0,D:2.0):3.0):0.0;",
                        "C",
                        "D",
                        8.0,
                        "A",
                        "B",
                        "C",
                        "D"
                ),
                new DistanceCase(
                        "five taxon pectinate increase",
                        "((((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):2.0,E:5.0):0.0;",
                        "A",
                        "B",
                        7.0,
                        "A",
                        "B",
                        "C",
                        "D",
                        "E"
                ),
                new DistanceCase(
                        "five taxon balanced decrease",
                        "(((A:1.0,B:1.0):3.0,C:4.0):2.0,(D:2.0,E:2.0):4.0):0.0;",
                        "A",
                        "E",
                        5.0,
                        "A",
                        "B",
                        "C",
                        "D",
                        "E"
                ),
                new DistanceCase(
                        "six taxon balanced increase",
                        "(((A:1.0,B:1.0):2.0,(C:1.0,D:1.0):2.0):2.0,(E:1.0,F:1.0):4.0):0.0;",
                        "A",
                        "B",
                        8.0,
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F"
                ),
                new DistanceCase(
                        "six taxon nested decrease",
                        "((((A:1.0,B:1.0):2.0,C:3.0):2.0,D:5.0):2.0,(E:2.0,F:2.0):5.0):0.0;",
                        "A",
                        "F",
                        6.0,
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F"
                )
        };

        return Stream.of(cases).flatMap(testCase ->
                LongStream.of(seeds).mapToObj(seed -> Arguments.of(
                        testCase.name,
                        seed,
                        testCase.newick,
                        testCase.firstTaxon,
                        testCase.secondTaxon,
                        testCase.targetDistance,
                        testCase.expectedLeafIds
                ))
        );
    }

    public Random getRandom() {
        return new Random(1);
    }

    private record DistanceCase(
            String name,
            String newick,
            String firstTaxon,
            String secondTaxon,
            double targetDistance,
            String... expectedLeafIds
    ) {
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

    private static double distance(Node nodeA, Node nodeB) {
        Node mrca = TreeUtils.getCommonAncestor(nodeA, nodeB).mrca();
        return 2.0 * mrca.getHeight() - nodeA.getHeight() - nodeB.getHeight();
    }

    private static void assertTreeIsConnected(Node root, String... expectedLeafIds) {
        Set<Node> visitedNodes = new HashSet<>();
        Set<String> leafIds = new HashSet<>();
        collectNodes(root, null, visitedNodes, leafIds);

        assertThat(leafIds).containsExactlyInAnyOrder(expectedLeafIds);
        assertThat(visitedNodes).hasSize(2 * expectedLeafIds.length - 1);
    }

    private static void collectNodes(Node node, Node expectedParent, Set<Node> visitedNodes, Set<String> leafIds) {
        assertThat(node.getParent()).isSameAs(expectedParent);
        assertThat(visitedNodes.add(node)).isTrue();

        if (node.isLeaf()) {
            leafIds.add(node.getID());
        }

        for (Node child : node.getChildren()) {
            collectNodes(child, node, visitedNodes, leafIds);
        }
    }

}
