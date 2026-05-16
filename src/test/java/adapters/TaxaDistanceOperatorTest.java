package adapters;

import adaptiveoperators.TaxaDistanceOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TaxaDistanceOperatorTest {

    @Test
    void initRejectsTreeWithFewerThanTwoTaxa() {
        TreeParser tree = parseTree("A:0.0;");
        TaxaDistanceOperator operator = new TaxaDistanceOperator();
        operator.treeInput.setValue(tree, operator);

        assertThatThrownBy(operator::initAndValidate)
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("at least two taxa");
    }

    @Test
    void initRejectsNonPositiveNumberOfPairs() {
        TreeParser tree = parseTree("(A:1.0,B:1.0):0.0;");
        TaxaDistanceOperator operator = new TaxaDistanceOperator();
        operator.treeInput.setValue(tree, operator);
        operator.numberOfPairsInput.setValue(0, operator);

        assertThatThrownBy(operator::initAndValidate)
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("numberOfPairs");
    }

    @Test
    void initCapsSampledPairsAtAvailableUnorderedPairs() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
        TaxaDistanceOperator operator = new TaxaDistanceOperator();

        operator.initByName("tree", tree, "numberOfPairs", 100, "weight", 1.0);

        assertThat(operator.getPairCount()).isEqualTo(3);
    }

    @Test
    void burnInProposalDoesNotRecordOrMutateTree() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
        TaxaDistanceOperator operator = initOperator(tree, 3);
        String originalNewick = tree.getRoot().toNewick();

        double hastingsRatio = operator.proposal();

        assertThat(hastingsRatio).isEqualTo(0.0);
        assertThat(operator.getProposalCount()).isEqualTo(1);
        for (int i = 0; i < operator.getPairCount(); i++) {
            assertThat(operator.getModelCount(i)).isZero();
        }
        assertThat(tree.getRoot().toNewick()).isEqualTo(originalNewick);
    }

    @Test
    void trainingProposalRecordsAllSampledPairDistancesWithoutMutation() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
        TaxaDistanceOperator operator = initOperator(tree, 3);
        operator.setProposalCount(1_999);
        String originalNewick = tree.getRoot().toNewick();

        double hastingsRatio = operator.proposal();

        assertThat(hastingsRatio).isEqualTo(0.0);
        for (int i = 0; i < operator.getPairCount(); i++) {
            assertThat(operator.getModelCount(i)).isEqualTo(1);
        }
        assertThat(tree.getRoot().toNewick()).isEqualTo(originalNewick);
    }

    @Test
    void activeProposalReturnsNegativeInfinityWhenModelHasTooFewSamples() {
        TreeParser tree = parseTree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
        TaxaDistanceOperator operator = initOperator(tree, 3);
        operator.setProposalCount(20_000);

        double hastingsRatio = operator.proposal();

        assertThat(hastingsRatio).isEqualTo(Double.NEGATIVE_INFINITY);
    }

    @Test
    void activeProposalChangesTreeAndReturnsFiniteHastingsRatio() {
        Randomizer.setSeed(1);
        TreeParser tree = parseTree("((A:1.0,B:1.0):2.0,(C:1.0,D:1.0):2.0):0.0;");
        TaxaDistanceOperator operator = initOperator(tree, 6);
        operator.setRandom(new Random(3));
        operator.setProposalCount(1_999);

        operator.proposal();
        stretchTree(tree);
        operator.proposal();
        String trainedNewick = tree.getRoot().toNewick();
        operator.setProposalCount(20_000);

        double hastingsRatio = operator.proposal();

        assertThat(hastingsRatio).isFinite();
        assertThat(tree.getRoot().toNewick()).isNotEqualTo(trainedNewick);
        assertTreeIsConnected(tree.getRoot(), "A", "B", "C", "D");
    }

    @Test
    void listStateNodesContainsTree() {
        TreeParser tree = parseTree("(A:1.0,B:1.0):0.0;");
        TaxaDistanceOperator operator = initOperator(tree, 1);

        assertThat(operator.listStateNodes()).containsExactly(tree);
    }

    private static TaxaDistanceOperator initOperator(TreeParser tree, int numberOfPairs) {
        TaxaDistanceOperator operator = new TaxaDistanceOperator();
        operator.initByName("tree", tree, "numberOfPairs", numberOfPairs, "weight", 1.0);
        return operator;
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

    private static void stretchTree(TreeParser tree) {
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                node.setHeight(node.getHeight() * 1.5);
            }
        }
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
