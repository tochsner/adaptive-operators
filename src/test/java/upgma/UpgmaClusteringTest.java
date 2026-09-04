package upgma;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Pins {@link UpgmaClustering} to the straightforward full-scan algorithm.
 *
 * <p>{@link #reference} is that algorithm written out plainly: rescan the whole working matrix for
 * the closest pair at every merge. Any faster search has to return the identical result, down to
 * which pair it picks out of a tie, since the sampled distance matrix and the tree it produces feed
 * straight into a Hastings ratio.
 */
class UpgmaClusteringTest {

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L, 4L, 5L})
    void matchesTheFullScanOnRandomMatrices(long seed) {
        Random random = new Random(seed);

        for (int leafCount = 2; leafCount <= 40; leafCount++) {
            double[][] distances = UpgmaFixtures.randomSymmetric(leafCount, random);
            assertSameClustering(distances, "leafCount = " + leafCount);
        }
    }

    /**
     * Ties are what separates a nearest-neighbour cache from the full scan, so they get their own
     * case: distances rounded onto a coarse grid produce plenty of exactly equal pairs.
     */
    @ParameterizedTest
    @ValueSource(longs = {11L, 12L, 13L, 14L, 15L})
    void matchesTheFullScanWhenDistancesTie(long seed) {
        Random random = new Random(seed);

        for (int leafCount = 3; leafCount <= 60; leafCount++) {
            // only a handful of distinct values, so equal pairs are the rule rather than the exception
            int values = 1 + random.nextInt(3);
            double[][] distances = new double[leafCount][leafCount];
            for (int i = 0; i < leafCount; i++) {
                for (int j = i + 1; j < leafCount; j++) {
                    double value = 1 + random.nextInt(values);
                    distances[i][j] = value;
                    distances[j][i] = value;
                }
            }
            assertSameClustering(distances, "leafCount = " + leafCount + ", values = " + values);
        }
    }

    @Test
    void handlesAMatrixWhereEveryDistanceIsEqual() {
        for (int leafCount = 2; leafCount <= 12; leafCount++) {
            double[][] distances = new double[leafCount][leafCount];
            for (int i = 0; i < leafCount; i++) {
                for (int j = i + 1; j < leafCount; j++) {
                    distances[i][j] = 1.0;
                    distances[j][i] = 1.0;
                }
            }
            assertSameClustering(distances, "leafCount = " + leafCount);
        }
    }

    private static void assertSameClustering(double[][] distances, String description) {
        UpgmaClustering.Result actual = UpgmaClustering.cluster(distances);
        UpgmaClustering.Result expected = reference(distances);

        assertThat(actual.mergeLeft()).as(description).isEqualTo(expected.mergeLeft());
        assertThat(actual.mergeRight()).as(description).isEqualTo(expected.mergeRight());
        assertThat(actual.mergeHeight()).as(description).isEqualTo(expected.mergeHeight());
    }

    /** Average-linkage clustering with a full rescan for the closest pair at every merge. */
    private static UpgmaClustering.Result reference(double[][] distances) {
        int n = distances.length;
        int mergeCount = n - 1;

        int[] mergeLeft = new int[mergeCount];
        int[] mergeRight = new int[mergeCount];
        double[] mergeHeight = new double[mergeCount];

        int activeCount = n;
        int[] id = new int[n];
        int[] size = new int[n];
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            id[i] = i;
            size[i] = 1;
            System.arraycopy(distances[i], 0, d[i], 0, n);
        }

        for (int merge = 0; merge < mergeCount; merge++) {
            int bestI = -1;
            int bestJ = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int i = 0; i < activeCount; i++) {
                for (int j = i + 1; j < activeCount; j++) {
                    if (d[i][j] < best) {
                        best = d[i][j];
                        bestI = i;
                        bestJ = j;
                    }
                }
            }

            mergeLeft[merge] = id[bestI];
            mergeRight[merge] = id[bestJ];
            mergeHeight[merge] = best / 2.0;

            int sizeI = size[bestI];
            int sizeJ = size[bestJ];
            int newSize = sizeI + sizeJ;

            for (int k = 0; k < activeCount; k++) {
                if (k == bestI || k == bestJ) {
                    continue;
                }
                double updated = (sizeI * d[bestI][k] + sizeJ * d[bestJ][k]) / newSize;
                d[bestI][k] = updated;
                d[k][bestI] = updated;
            }
            id[bestI] = n + merge;
            size[bestI] = newSize;

            int last = activeCount - 1;
            if (bestJ != last) {
                id[bestJ] = id[last];
                size[bestJ] = size[last];
                for (int k = 0; k < last; k++) {
                    d[bestJ][k] = d[last][k];
                    d[k][bestJ] = d[k][last];
                }
            }
            activeCount--;
        }

        return new UpgmaClustering.Result(mergeLeft, mergeRight, mergeHeight);
    }
}
