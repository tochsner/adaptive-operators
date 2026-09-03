package distancematrix;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.evolution.operator.Exchange;
import beast.base.evolution.operator.Uniform;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Checks that {@link DistanceMatrixOperator} actually leaves the tree posterior invariant.
 *
 * <p>The target is chosen so that the answer is known in closed form. Taking
 * {@code pi(T) proportional to exp(-lambda * sum_v w_v)} over ranked labelled topologies and ordered
 * node distances {@code w_v = 2 * h_v} makes the topology uniform over the
 * {@code n! (n-1)! / 2^(n-1)} ranked labelled topologies, and makes the node distances the order
 * statistics of {@code n - 1} independent exponentials. Both are checked against the sampled chain.
 *
 * <p>The chain here is a plain Metropolis-Hastings loop rather than a BEAST {@code MCMC}, so that
 * the test can store and restore the tree itself without standing up a full {@code State}. The
 * operator, its Hastings ratio and its accept and reject bookkeeping are the real ones.
 */
class DistanceMatrixOperatorInvarianceTest {

    private static final int LEAF_COUNT = 4;
    private static final double HEIGHT_RATE = 1.0;

    private static final int BURN_IN = 200_000;
    private static final int SAMPLE_COUNT = 2_000_000;

    // biasing the choice of entry towards the spanning tree mixes the topology faster but leaves
    // more autocorrelation in the sampled frequencies, so a few chains are pooled
    private static final int CHAIN_COUNT = 3;

    /** Ranked labelled topologies on four taxa: 4! 3! / 2^3. */
    private static final int RANKED_TOPOLOGY_COUNT = 18;

    @Test
    void samplesTheTargetWhenTheFibreVolumeIsCorrected() {
        Map<String, Integer> topologyCounts = new HashMap<>();
        double[] meanDistance = new double[LEAF_COUNT - 1];

        for (int chainNr = 0; chainNr < CHAIN_COUNT; chainNr++) {
            Randomizer.setSeed(20240830L + chainNr);
            Chain chain = run(true, false);

            for (Map.Entry<String, Integer> entry : chain.topologyCounts.entrySet()) {
                topologyCounts.merge(entry.getKey(), entry.getValue(), Integer::sum);
            }
            for (int rank = 0; rank < meanDistance.length; rank++) {
                meanDistance[rank] += chain.meanDistance[rank] / CHAIN_COUNT;
            }
        }

        assertThat(topologyCounts).hasSize(RANKED_TOPOLOGY_COUNT);

        double expectedFrequency = 1.0 / RANKED_TOPOLOGY_COUNT;
        long total = (long) SAMPLE_COUNT * CHAIN_COUNT;
        for (Map.Entry<String, Integer> entry : topologyCounts.entrySet()) {
            assertThat(entry.getValue() / (double) total)
                    .as("frequency of ranked topology " + entry.getKey())
                    .isCloseTo(expectedFrequency, within(0.008));
        }

        // the node distances are the order statistics of three independent Exp(lambda) variables
        assertThat(meanDistance[0]).as("smallest node distance")
                .isCloseTo(1.0 / (3.0 * HEIGHT_RATE), within(0.02));
        assertThat(meanDistance[1]).as("middle node distance")
                .isCloseTo((1.0 / 3.0 + 1.0 / 2.0) / HEIGHT_RATE, within(0.03));
        assertThat(meanDistance[2]).as("largest node distance")
                .isCloseTo((1.0 / 3.0 + 1.0 / 2.0 + 1.0) / HEIGHT_RATE, within(0.05));
    }

    @Test
    void samplesTheTargetWhenCombinedWithStandardTreeOperators() {
        // other tree operators know nothing about the matrix, so they break the invariant that the
        // tree is the clustering of the matrix. The operator repairs that by redrawing the matrix
        // from its exact conditional given the tree, which is a Gibbs step and so leaves the target
        // alone. This checks that the combination really does still sample the target.
        Map<String, Integer> topologyCounts = new HashMap<>();

        for (int chainNr = 0; chainNr < 2; chainNr++) {
            Randomizer.setSeed(600613L + chainNr);
            Chain chain = run(true, true);

            for (Map.Entry<String, Integer> entry : chain.topologyCounts.entrySet()) {
                topologyCounts.merge(entry.getKey(), entry.getValue(), Integer::sum);
            }
        }

        assertThat(topologyCounts).hasSize(RANKED_TOPOLOGY_COUNT);

        long total = (long) SAMPLE_COUNT * 2;
        for (Map.Entry<String, Integer> entry : topologyCounts.entrySet()) {
            assertThat(entry.getValue() / (double) total)
                    .as("frequency of ranked topology " + entry.getKey())
                    .isCloseTo(1.0 / RANKED_TOPOLOGY_COUNT, within(0.008));
        }
    }

    @Test
    void missesTheTargetWhenTheFibreVolumeIsIgnored() {
        Randomizer.setSeed(20240830L);
        Chain chain = run(false, false);

        // without the correction the walk is free to drift along the fibres of the matrix to tree
        // map, so the tree it induces is not distributed as the target
        double totalExpected = (1.0 / 3.0 + 2.0 * (1.0 / 3.0 + 1.0 / 2.0)
                + (1.0 / 3.0 + 1.0 / 2.0 + 1.0)) / HEIGHT_RATE;
        double totalSampled = chain.meanDistance[0] + chain.meanDistance[1] + chain.meanDistance[2];

        assertThat(Math.abs(totalSampled - totalExpected))
                .as("the uncorrected chain is measurably biased")
                .isGreaterThan(0.1);
    }

    private static Chain run(boolean correctFibreVolume, boolean withOtherTreeOperators) {
        TreeParser tree = parseTree("(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0):0.0;");

        DistanceMatrixOperator operator = new DistanceMatrixOperator();
        operator.initByName(
                "tree", tree,
                "weight", 1.0,
                "referenceDensity", "exponential",
                "correctFibreVolume", correctFibreVolume);

        // the operator's accept and reject bookkeeping reads its schedule, so give it one
        OperatorSchedule schedule = new OperatorSchedule();
        schedule.initAndValidate();
        operator.setOperatorSchedule(schedule);

        List<Operator> operators = new ArrayList<>();
        operators.add(operator);

        if (withOtherTreeOperators) {
            Exchange narrow = new Exchange();
            narrow.initByName("tree", tree, "weight", 1.0);
            narrow.setOperatorSchedule(schedule);
            operators.add(narrow);

            Uniform uniform = new Uniform();
            uniform.initByName("tree", tree, "weight", 1.0);
            uniform.setOperatorSchedule(schedule);
            operators.add(uniform);
        }

        Chain chain = new Chain();
        TreeSnapshot snapshot = new TreeSnapshot(tree);
        double logTarget = logTarget(tree);

        for (int iteration = 0; iteration < BURN_IN + SAMPLE_COUNT; iteration++) {
            snapshot.store(tree);

            Operator chosen = operators.get(Randomizer.nextInt(operators.size()));
            double logHastingsRatio = chosen.proposal();

            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                chosen.reject(-1);
                snapshot.restore(tree);
            } else {
                double proposedLogTarget = logTarget(tree);
                double logAlpha = proposedLogTarget - logTarget + logHastingsRatio;

                if (Math.log(Randomizer.nextDouble()) < logAlpha) {
                    chosen.accept();
                    logTarget = proposedLogTarget;
                } else {
                    chosen.reject(0);
                    snapshot.restore(tree);
                }
            }

            if (iteration >= BURN_IN) {
                chain.record(tree);
            }
        }

        for (int rank = 0; rank < LEAF_COUNT - 1; rank++) {
            chain.meanDistance[rank] /= SAMPLE_COUNT;
        }

        return chain;
    }

    /** Exponential on the node distances, flat over ranked topologies, up to a constant. */
    private static double logTarget(Tree tree) {
        double sumOfDistances = 0.0;

        for (Node node : sortedInternalNodes(tree)) {
            sumOfDistances += 2.0 * node.getHeight();
        }

        return -HEIGHT_RATE * sumOfDistances;
    }

    private static List<Node> sortedInternalNodes(Tree tree) {
        List<Node> internal = new ArrayList<>();

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            if (!node.isLeaf()) {
                internal.add(node);
            }
        }

        internal.sort(Comparator.comparingDouble(Node::getHeight));
        return internal;
    }

    private static class Chain {
        private final Map<String, Integer> topologyCounts = new HashMap<>();
        private final double[] meanDistance = new double[LEAF_COUNT - 1];

        private void record(Tree tree) {
            List<Node> internal = sortedInternalNodes(tree);
            StringJoiner topology = new StringJoiner("|");

            for (int rank = 0; rank < internal.size(); rank++) {
                Node node = internal.get(rank);
                this.meanDistance[rank] += 2.0 * node.getHeight();
                topology.add(cladeOf(node));
            }

            this.topologyCounts.merge(topology.toString(), 1, Integer::sum);
        }

        /** Sorted leaf identifiers below a node. In height order these name a ranked topology. */
        private static String cladeOf(Node node) {
            List<String> leaves = new ArrayList<>();
            collectLeaves(node, leaves);
            leaves.sort(Comparator.naturalOrder());
            return String.join(",", leaves);
        }

        private static void collectLeaves(Node node, List<String> leaves) {
            if (node.isLeaf()) {
                leaves.add(node.getID());
                return;
            }
            for (Node child : node.getChildren()) {
                collectLeaves(child, leaves);
            }
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
