package upgma;

/**
 * Plain average-linkage (UPGMA) clustering of a fixed distance matrix.
 *
 * <p>Repeatedly merges the two closest active clusters, updating the remaining cluster distances
 * with the usual weighted-average (Lance-Williams) rule
 * {@code d(X, k) = (|A| d(A, k) + |B| d(B, k)) / |X|}, until a single cluster remains.
 *
 * <p>The {@code n - 1} merges are reported in the {@code (mergeLeft, mergeRight, mergeHeight)} form
 * {@code adapters.TreeUtils#applyClustering} expects: leaves are clusters {@code 0 .. n-1}, matching
 * the input matrix's own indexing, and the cluster created by the {@code k}-th merge is
 * {@code n + k}, the same numbering {@code distancematrix.SingleLinkageTree} uses. Merge heights are
 * half the merge distance, following the standard UPGMA convention that a node's height is the
 * cophenetic distance divided by two.
 */
public final class UpgmaClustering {

    private UpgmaClustering() {
    }

    public static final class Result {
        public final int[] mergeLeft;
        public final int[] mergeRight;
        public final double[] mergeHeight;

        private Result(int[] mergeLeft, int[] mergeRight, double[] mergeHeight) {
            this.mergeLeft = mergeLeft;
            this.mergeRight = mergeRight;
            this.mergeHeight = mergeHeight;
        }
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
        }

        return new Result(mergeLeft, mergeRight, mergeHeight);
    }
}
