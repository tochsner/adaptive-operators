package upgma;

import beast.base.evolution.tree.Tree;
import beast.base.util.GammaFunction;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;
import java.util.SplittableRandom;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Pins {@link UpgmaConditionalDensity} to the cluster distances read straight off D.
 *
 * <p>{@link #reference} averages over leaf pairs for every cluster distance it needs, the way the
 * definition reads. Propagating those averages along the merges instead sums the same numbers in a
 * different order, so the two agree to rounding rather than exactly.
 */
class UpgmaConditionalDensityTest {

    private static final double CONCENTRATION = 100.0;

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L, 4L, 5L})
    void matchesTheLeafPairDefinition(long seed) {
        Random random = new Random(seed);
        SplittableRandom rng = new SplittableRandom(seed);

        for (int leafCount = 3; leafCount <= 25; leafCount++) {
            Tree tree = UpgmaFixtures.randomTree(leafCount, random);
            UpgmaSplitModel model = new UpgmaSplitModel(
                    tree,
                    UpgmaFixtures.randomSymmetric(leafCount, random),
                    new double[leafCount],
                    UpgmaFixtures.rate(1.0),
                    CONCENTRATION);

            // a matrix the density is actually meant to score, i.e. one that clusters back to the tree
            double[][] distances = new double[leafCount][leafCount];
            new UpgmaDistanceSampler(leafCount, rng).sample(model, distances);

            assertThat(new UpgmaConditionalDensity(leafCount).logDensity(model, distances))
                    .as("leafCount = " + leafCount)
                    .isCloseTo(reference(model, UpgmaFixtures.clades(tree), distances), within(1e-9));
        }
    }

    /** {@link UpgmaConditionalDensity#logDensity} with every cluster distance averaged over leaf pairs. */
    private static double reference(UpgmaSplitModel model, int[][] clades, double[][] distances) {
        double logDensity = 0.0;

        for (UpgmaSplitModel.Split split : model.splits()) {
            double dr = model.mergeDistance(split.r());

            int[] active = split.active();
            for (int i = 0; i < active.length; i++) {
                double dVA = referenceDistance(clades, distances, active[i], split.leftNr());
                double dVB = referenceDistance(clades, distances, active[i], split.rightNr());
                double dVX = (split.leftSize() * dVA + split.rightSize() * dVB) / split.size();

                double base = split.size() * (dVX - dr);
                double p = split.leftSize() * (dVA - dr) / base;

                double mean = model.betaMean(split, i);
                logDensity += logBetaPdf(p, model.alpha(mean), model.beta(mean));
                logDensity -= Math.log(split.size()) + Math.log(base)
                        - Math.log(split.leftSize()) - Math.log(split.rightSize());
            }
        }

        return logDensity;
    }

    private static double referenceDistance(int[][] clades, double[][] distances, int nodeV, int nodeA) {
        double sum = 0.0;
        for (int v : clades[nodeV]) {
            for (int a : clades[nodeA]) {
                sum += distances[v][a];
            }
        }
        return sum / (clades[nodeV].length * (double) clades[nodeA].length);
    }

    private static double logBetaPdf(double p, double alpha, double beta) {
        double logNormalizer =
                GammaFunction.lnGamma(alpha) + GammaFunction.lnGamma(beta) - GammaFunction.lnGamma(alpha + beta);
        return (alpha - 1.0) * Math.log(p) + (beta - 1.0) * Math.log(1.0 - p) - logNormalizer;
    }
}
