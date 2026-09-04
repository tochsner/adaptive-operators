package upgma;

/**
 * Plain average-linkage (UPGMA) clustering of a fixed distance matrix: repeatedly merges the two
 * closest clusters under {@code d(X, k) = (|A| d(A, k) + |B| d(B, k)) / |X|} until one remains.
 *
 * <p>The {@code n - 1} merges come back in the form {@code adapters.TreeUtils#applyClustering}
 * expects: leaves are clusters {@code 0 .. n-1} and the {@code k}-th merge creates cluster
 * {@code n + k}. Merge heights are half the merge distance, the usual cophenetic convention.
 *
 * <p>Each active slot caches its nearest neighbour to the right, so finding the closest pair is a
 * scan over that cache rather than over the whole matrix. Restricting the cache to {@code j > i}
 * makes it pick the same pair as a plain row-major scan would, ties included, which matters because
 * the tree this produces feeds a Hastings ratio. A merge only invalidates the rows whose neighbour
 * it touched, so the pass costs {@code O(n^2)} on distances that are not systematically tied.
 */
public final class UpgmaClustering {

    private UpgmaClustering() {
    }

    public record Result(int[] mergeLeft, int[] mergeRight, double[] mergeHeight) {
    }

    /**
     * @param distances symmetric {@code n x n} distance matrix, indexed like the leaves it describes
     */
    public static Result cluster(double[][] distances) {
        int n = distances.length;
        int mergeCount = n - 1;

        int[] mergeLeft = new int[mergeCount];
        int[] mergeRight = new int[mergeCount];
        double[] mergeHeight = new double[mergeCount];

        // dense, reindexed working copy: slot i initially holds leaf i, and both a merged cluster's
        // id and its slot's contents get overwritten in place as clusters merge and slots compact
        int activeCount = n;
        int[] id = new int[n];
        int[] size = new int[n];
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            id[i] = i;
            size[i] = 1;
            System.arraycopy(distances[i], 0, d[i], 0, n);
        }

        // nearest[i] is the smallest j > i minimising d[i][j], and nearestDistance[i] that minimum
        int[] nearest = new int[n];
        double[] nearestDistance = new double[n];
        for (int i = 0; i < activeCount; i++) {
            findNearest(d, i, activeCount, nearest, nearestDistance);
        }

        for (int merge = 0; merge < mergeCount; merge++) {
            int bestI = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int i = 0; i < activeCount; i++) {
                if (nearestDistance[i] < best) {
                    best = nearestDistance[i];
                    bestI = i;
                }
            }
            int bestJ = nearest[bestI];

            mergeLeft[merge] = id[bestI];
            mergeRight[merge] = id[bestJ];
            mergeHeight[merge] = best / 2.0;

            int sizeI = size[bestI];
            int sizeJ = size[bestJ];
            int newSize = sizeI + sizeJ;

            // fold bestJ into bestI's slot, which now stands for the merged cluster
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

            // compact away bestJ's now-stale slot by moving the last active slot into it
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

            refreshNearest(d, activeCount, bestI, bestJ, last, nearest, nearestDistance);
        }

        return new Result(mergeLeft, mergeRight, mergeHeight);
    }

    /**
     * Brings the cache back in step with the matrix after a merge of {@code bestI} and {@code bestJ}
     * (with {@code bestI < bestJ}) and the compaction that moved slot {@code last} into
     * {@code bestJ}.
     *
     * <p>Only two slots hold something new: {@code bestI}, which is the merged cluster, and
     * {@code bestJ}, which is whatever was moved into it. Every other row keeps its cached
     * neighbour, unless that neighbour was one of the slots that just changed, in which case there
     * is nothing to compare against and the row is scanned again.
     */
    private static void refreshNearest(
            double[][] d, int activeCount, int bestI, int bestJ, int last,
            int[] nearest, double[] nearestDistance) {
        for (int i = 0; i < activeCount; i++) {
            if (i == bestI || i == bestJ) {
                findNearest(d, i, activeCount, nearest, nearestDistance);
                continue;
            }

            // the merged cluster is gone from bestJ, and whatever sat at last is no longer there
            if (nearest[i] == bestI || nearest[i] == bestJ || nearest[i] == last) {
                findNearest(d, i, activeCount, nearest, nearestDistance);
                continue;
            }

            // the cached neighbour is untouched, so only the two changed slots can beat it
            if (bestI > i) {
                offer(d[i][bestI], bestI, i, nearest, nearestDistance);
            }
            if (bestJ != last && bestJ > i && bestJ < activeCount) {
                offer(d[i][bestJ], bestJ, i, nearest, nearestDistance);
            }
        }
    }

    /** Takes {@code candidate} as row {@code i}'s neighbour if a row-major scan would prefer it. */
    private static void offer(
            double distance, int candidate, int i, int[] nearest, double[] nearestDistance) {
        if (distance < nearestDistance[i] || (distance == nearestDistance[i] && candidate < nearest[i])) {
            nearestDistance[i] = distance;
            nearest[i] = candidate;
        }
    }

    /** Scans row {@code i} for the first {@code j > i} attaining its smallest distance. */
    private static void findNearest(
            double[][] d, int i, int activeCount, int[] nearest, double[] nearestDistance) {
        int bestJ = -1;
        double best = Double.POSITIVE_INFINITY;
        for (int j = i + 1; j < activeCount; j++) {
            if (d[i][j] < best) {
                best = d[i][j];
                bestJ = j;
            }
        }
        nearest[i] = bestJ;
        nearestDistance[i] = best;
    }
}
