package upgma;

import beast.base.evolution.tree.Tree;
import beast.base.spec.type.RealScalar;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Pins {@link UpgmaSplitModel#betaMean} to its definition in {@code approach.md}, evaluated
 * directly over leaf pairs.
 *
 * <p>{@link #referenceMean} averages the substitution scores over the leaves of the two clades, the
 * way the definition reads. Propagating those averages along the merges instead is much cheaper but
 * sums the same numbers in a different order, so the two agree to rounding rather than exactly.
 */
class UpgmaSplitModelTest {

    private static final double CONCENTRATION = 100.0;
    private static final double MIN_MEAN = 0.1;
    private static final double EXCESS_FLOOR = 0.1;

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L, 4L, 5L})
    void betaMeansMatchTheLeafPairDefinition(long seed) {
        Random random = new Random(seed);

        for (int leafCount = 3; leafCount <= 25; leafCount++) {
            Tree tree = UpgmaFixtures.randomTree(leafCount, random);
            double[][] substitutionDistances = UpgmaFixtures.randomSymmetric(leafCount, random);
            double[] tipDates = new double[leafCount];
            for (int leaf = 0; leaf < leafCount; leaf++) {
                tipDates[leaf] = 0.1 * random.nextDouble();
            }
            RealScalar<?> rate = UpgmaFixtures.rate(0.3 + random.nextDouble());

            UpgmaSplitModel model =
                    new UpgmaSplitModel(tree, substitutionDistances, tipDates, rate, CONCENTRATION);
            int[][] clades = UpgmaFixtures.clades(tree);

            for (UpgmaSplitModel.Split split : model.splits()) {
                int[] active = split.active();
                for (int i = 0; i < active.length; i++) {
                    double expected = referenceMean(
                            clades, substitutionDistances, tipDates, rate.get(), active[i], split);

                    assertThat(model.betaMean(split, i))
                            .as("leafCount = " + leafCount + ", split " + split.nodeNr()
                                    + ", v " + active[i])
                            .isCloseTo(expected, within(1e-12));
                }
            }
        }
    }

    /** {@link UpgmaSplitModel#betaMean} with every score averaged straight over the leaf pairs. */
    private static double referenceMean(
            int[][] clades, double[][] substitutionDistances, double[] tipDates, double rate,
            int nodeV, UpgmaSplitModel.Split split) {
        double splitScore = referenceScore(
                clades, substitutionDistances, tipDates, rate, split.leftNr(), split.rightNr());
        double scoreVA =
                referenceScore(clades, substitutionDistances, tipDates, rate, nodeV, split.leftNr());
        double scoreVB =
                referenceScore(clades, substitutionDistances, tipDates, rate, nodeV, split.rightNr());

        double baseline = Math.min(splitScore, (1.0 - EXCESS_FLOOR) * Math.min(scoreVA, scoreVB));
        double weightedA = split.leftSize() * Math.max(scoreVA - baseline, 0.0);
        double weightedB = split.rightSize() * Math.max(scoreVB - baseline, 0.0);
        double total = weightedA + weightedB;

        double mean = total > 0.0 ? weightedA / total : split.leftSize() / (double) split.size();
        return Math.min(Math.max(mean, MIN_MEAN), 1.0 - MIN_MEAN);
    }

    private static double referenceScore(
            int[][] clades, double[][] substitutionDistances, double[] tipDates, double rate,
            int nodeV, int nodeA) {
        double sum = 0.0;
        for (int v : clades[nodeV]) {
            for (int a : clades[nodeA]) {
                sum += substitutionDistances[v][a] / rate + tipDates[v] + tipDates[a];
            }
        }
        return sum / (clades[nodeV].length * (double) clades[nodeA].length);
    }
}
