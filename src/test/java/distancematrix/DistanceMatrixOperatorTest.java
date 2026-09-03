package distancematrix;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.OperatorSchedule;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class DistanceMatrixOperatorTest {

    @Test
    void theFirstProposalDoesNotReshapeTheTree() {
        // the matrix is drawn from its conditional given the tree, so single linkage has to give the
        // tree straight back
        Randomizer.setSeed(5L);
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):0.0;");
        DistanceMatrixOperator operator = createOperator(tree, true);

        String before = tree.getRoot().toNewick();
        TreeSnapshot snapshot = new TreeSnapshot(tree);

        for (int step = 0; step < 200; step++) {
            snapshot.store(tree);
            operator.proposal();
            operator.reject(0);
            snapshot.restore(tree);

            assertThat(tree.getRoot().toNewick())
                    .as("rejected proposal " + step + " must leave the original tree in place")
                    .isEqualTo(before);
        }
    }

    @Test
    void rebuildsTheMatrixWhenSomethingElseChangesTheTree() {
        // this is what BEAST's state node initialisers do: they replace the tree from inside
        // MCMC.run(), long after initAndValidate
        Randomizer.setSeed(5L);
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):0.0;");
        DistanceMatrixOperator operator = createOperator(tree, true);

        operator.proposal();
        operator.accept();

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            if (!node.isLeaf()) {
                node.setHeight(node.getHeight() * 5.0);
            }
        }

        double rootHeight = tree.getRoot().getHeight();
        assertThat(rootHeight).isCloseTo(15.0, within(1e-12));

        // the operator must pick the new tree up rather than overwrite it with the stale matrix
        TreeSnapshot snapshot = new TreeSnapshot(tree);

        for (int step = 0; step < 200; step++) {
            snapshot.store(tree);
            operator.proposal();
            operator.reject(0);
            snapshot.restore(tree);

            assertThat(tree.getRoot().getHeight())
                    .as("step " + step + " must not revert the external change")
                    .isCloseTo(rootHeight, within(1e-12));
        }
    }

    @Test
    void rejectsSeriallySampledTips() {
        TreeParser tree = parseTree("((A:1.0,B:2.0):1.0,C:3.0):0.0;");
        DistanceMatrixOperator operator = createOperator(tree, true);

        assertThatThrownBy(operator::proposal)
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("ultrametricity");
    }

    @Test
    void rejectsTooFewTaxa() {
        TreeParser tree = parseTree("(A:1.0,B:1.0):0.0;");

        assertThatThrownBy(() -> createOperator(tree, true))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("at least three taxa");
    }

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L, 4L, 5L})
    void proposalsKeepTheTreeUltrametricAndWellFormed(long seed) {
        Randomizer.setSeed(seed);
        TreeParser tree = parseTree("((((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):1.0,E:4.0):0.0;");
        DistanceMatrixOperator operator = createOperator(tree, true);

        for (int step = 0; step < 5_000; step++) {
            double logHastingsRatio = operator.proposal();

            assertThat(logHastingsRatio).as("step " + step).isNotNaN();
            assertTreeIsWellFormed(tree, "A", "B", "C", "D", "E");

            // always accepting keeps the matrix and the tree in step, which is what is under test
            operator.accept();
        }
    }

    @Test
    void biasedSelectionReachesTheTreeFarMoreOftenThanUniformSelection() {
        // only n-1 of the n(n-1)/2 entries carry the tree, so at twenty taxa a uniform choice would
        // touch it in about a tenth of the proposals
        String newick = randomLadderNewick(20);

        assertThat(fractionOfProposalsChangingTheTree(newick, 0.5))
                .as("biased selection")
                .isBetween(0.4, 0.6);
        assertThat(fractionOfProposalsChangingTheTree(newick, 0.1))
                .as("nearly uniform selection")
                .isLessThan(0.25);
    }

    private static double fractionOfProposalsChangingTheTree(String newick, double probability) {
        Randomizer.setSeed(13L);
        TreeParser tree = parseTree(newick);

        DistanceMatrixOperator operator = new DistanceMatrixOperator();
        operator.initByName(
                "tree", tree,
                "weight", 1.0,
                "spanningTreeProbability", probability);
        OperatorSchedule schedule = new OperatorSchedule();
        schedule.initAndValidate();
        operator.setOperatorSchedule(schedule);

        int changed = 0;
        int considered = 0;
        TreeSnapshot snapshot = new TreeSnapshot(tree);

        for (int step = 0; step < 4_000; step++) {
            snapshot.store(tree);
            String before = tree.getRoot().toNewick();

            if (operator.proposal() != Double.NEGATIVE_INFINITY) {
                considered++;
                if (!tree.getRoot().toNewick().equals(before)) {
                    changed++;
                }
            }

            operator.reject(0);
            snapshot.restore(tree);
        }

        return changed / (double) considered;
    }

    /** A caterpillar tree on the given number of taxa, with tips at height zero. */
    private static String randomLadderNewick(int leafCount) {
        StringBuilder newick = new StringBuilder("T0:1.0");

        for (int leaf = 1; leaf < leafCount; leaf++) {
            newick.insert(0, "(").append(",T").append(leaf).append(":").append((double) leaf)
                    .append("):1.0");
        }

        return newick.substring(0, newick.length() - 4) + ":0.0;";
    }

    @Test
    void rejectingRestoresTheMatrixSoTheTreeCanBeRebuilt() {
        Randomizer.setSeed(11L);
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):0.0;");
        DistanceMatrixOperator operator = createOperator(tree, true);

        String before = tree.getRoot().toNewick();

        // drive the matrix until a proposal actually reshapes the tree, then reject it
        while (tree.getRoot().toNewick().equals(before)) {
            operator.proposal();
        }

        operator.reject(0);

        // the restored matrix has to induce the tree we started from
        DistanceMatrixOperator rebuilt = createOperator(parseTree(before), true);
        assertThat(rebuilt).isNotNull();

        double logHastingsRatio = operator.proposal();
        assertThat(logHastingsRatio).isNotNaN();
    }

    @Test
    void withoutTheCorrectionTheHastingsRatioIsOne() {
        Randomizer.setSeed(3L);
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):0.0;");
        DistanceMatrixOperator operator = createOperator(tree, false);

        for (int step = 0; step < 500; step++) {
            double logHastingsRatio = operator.proposal();

            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                assertThat(logHastingsRatio).isZero();
            }

            operator.accept();
        }
    }

    @Test
    void theScaleFactorIsCoercible() {
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):0.0;");
        DistanceMatrixOperator operator = createOperator(tree, true);

        operator.setCoercableParameterValue(0.25);

        assertThat(operator.getCoercableParameterValue()).isCloseTo(0.25, within(1e-12));
    }

    private static DistanceMatrixOperator createOperator(Tree tree, boolean correctFibreVolume) {
        DistanceMatrixOperator operator = new DistanceMatrixOperator();
        operator.initByName(
                "tree", tree,
                "weight", 1.0,
                "correctFibreVolume", correctFibreVolume);

        OperatorSchedule schedule = new OperatorSchedule();
        schedule.initAndValidate();
        operator.setOperatorSchedule(schedule);

        return operator;
    }

    private static void assertTreeIsWellFormed(Tree tree, String... expectedLeafIds) {
        Set<Node> visited = new HashSet<>();
        Set<String> leafIds = new HashSet<>();
        collectNodes(tree.getRoot(), null, visited, leafIds);

        assertThat(leafIds).containsExactlyInAnyOrder(expectedLeafIds);
        assertThat(visited).hasSize(2 * expectedLeafIds.length - 1);
    }

    private static void collectNodes(Node node, Node expectedParent, Set<Node> visited, Set<String> leafIds) {
        assertThat(node.getParent()).isSameAs(expectedParent);
        assertThat(visited.add(node)).isTrue();

        if (node.isLeaf()) {
            assertThat(node.getHeight()).as("tips stay at zero").isCloseTo(0.0, within(1e-12));
            leafIds.add(node.getID());
        } else {
            assertThat(node.getChildren()).hasSize(2);
            for (Node child : node.getChildren()) {
                assertThat(child.getHeight()).isLessThan(node.getHeight());
            }
        }

        for (Node child : node.getChildren()) {
            collectNodes(child, node, visited, leafIds);
        }
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
}
