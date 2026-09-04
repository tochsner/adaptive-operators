package upgma;

import beast.base.evolution.tree.Tree;
import beast.base.spec.type.RealScalar;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;
import java.util.SplittableRandom;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The invariant the whole augmentation rests on: a distance matrix drawn conditionally on a tree
 * has to cluster back to that very tree, so that {@code UPGMA(D) = T} and the conditional density
 * the Hastings ratio uses is evaluated on the tree it was meant for.
 */
class UpgmaDistanceSamplerTest {

    private static final double CONCENTRATION = 100.0;

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L, 4L, 5L})
    void sampledDistancesClusterBackToTheTree(long seed) {
        Random random = new Random(seed);
        SplittableRandom rng = new SplittableRandom(seed);

        for (int leafCount : new int[]{2, 3, 5, 20, 100}) {
            Tree tree = UpgmaFixtures.randomTree(leafCount, random);
            UpgmaSplitModel model = new UpgmaSplitModel(
                    tree,
                    UpgmaFixtures.randomSymmetric(leafCount, random),
                    new double[leafCount],
                    UpgmaFixtures.rate(1.0),
                    CONCENTRATION);

            double[][] distances = new double[leafCount][leafCount];
            new UpgmaDistanceSampler(model, rng).sample(distances);

            assertThat(UpgmaFixtures.cladesWithHeights(UpgmaClustering.cluster(distances), leafCount))
                    .as("leafCount = " + leafCount)
                    .isEqualTo(UpgmaFixtures.cladesWithHeights(tree));
        }
    }

    /** Distances have to be symmetric with a zero diagonal, and non-negative for UPGMA to read. */
    @ParameterizedTest
    @ValueSource(longs = {11L, 12L, 13L})
    void sampledDistancesAreSymmetricAndNonNegative(long seed) {
        Random random = new Random(seed);
        SplittableRandom rng = new SplittableRandom(seed);

        for (int leafCount : new int[]{2, 4, 15}) {
            Tree tree = UpgmaFixtures.randomTree(leafCount, random);
            UpgmaSplitModel model = new UpgmaSplitModel(
                    tree,
                    UpgmaFixtures.randomSymmetric(leafCount, random),
                    new double[leafCount],
                    UpgmaFixtures.rate(1.0),
                    CONCENTRATION);

            double[][] distances = new double[leafCount][leafCount];
            new UpgmaDistanceSampler(model, rng).sample(distances);

            for (int i = 0; i < leafCount; i++) {
                assertThat(distances[i][i]).isEqualTo(0.0);
                for (int j = 0; j < leafCount; j++) {
                    assertThat(distances[i][j]).isEqualTo(distances[j][i]);
                    assertThat(distances[i][j]).isGreaterThanOrEqualTo(0.0);
                }
            }
        }
    }
}
