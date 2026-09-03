package distancematrix;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SingleLinkageTreeTest {

    /** One merge, described independently of how the clustering happens to order the children. */
    private record Merge(BitSet clade, BitSet smallerChild, double height) {
    }

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L, 4L, 5L})
    void matchesBruteForceClustering(long seed) {
        Random random = new Random(seed);

        for (int leafCount = 3; leafCount <= 8; leafCount++) {
            double[] distances = randomDistances(leafCount, random);
            SingleLinkageTree clustering = new SingleLinkageTree(leafCount, distances);

            assertThat(mergesOf(clustering))
                    .as("leafCount = " + leafCount)
                    .isEqualTo(bruteForceMerges(leafCount, distances));
        }
    }

    @ParameterizedTest
    @ValueSource(longs = {11L, 12L, 13L, 14L, 15L})
    void reproducesAnUltrametricMatrixExactly(long seed) {
        Random random = new Random(seed);

        for (int leafCount = 3; leafCount <= 8; leafCount++) {
            double[] distances = randomDistances(leafCount, random);
            SingleLinkageTree clustering = new SingleLinkageTree(leafCount, distances);

            // the cophenetic matrix of the induced tree is ultrametric, so clustering it again has
            // to give back the very same tree
            double[] cophenetic = copheneticMatrix(clustering);
            SingleLinkageTree reclustered = new SingleLinkageTree(leafCount, cophenetic);

            assertThat(mergesOf(reclustered))
                    .as("leafCount = " + leafCount)
                    .isEqualTo(mergesOf(clustering));
            assertThat(copheneticMatrix(reclustered)).isEqualTo(cophenetic);
        }
    }

    @ParameterizedTest
    @ValueSource(longs = {21L, 22L, 23L, 24L, 25L})
    void incrementalUpdatesAgreeWithAFullRebuild(long seed) {
        Random random = new Random(seed);
        int leafCount = 7;

        double[] distances = randomDistances(leafCount, random);
        SingleLinkageTree clustering = new SingleLinkageTree(leafCount, distances);

        for (int step = 0; step < 2_000; step++) {
            int first = random.nextInt(leafCount);
            int second = random.nextInt(leafCount);
            if (first == second) {
                continue;
            }

            double newDistance = Math.abs(distances[clustering.pairIndex(first, second)]
                    + 0.4 * random.nextGaussian()) + 1e-6;
            distances[clustering.pairIndex(first, second)] = newDistance;

            List<Merge> before = mergesOf(clustering);
            SingleLinkageTree.UpdateKind kind = clustering.update(first, second, newDistance);
            List<Merge> after = mergesOf(clustering);

            assertThat(after)
                    .as("step " + step)
                    .isEqualTo(bruteForceMerges(leafCount, distances));
            assertThat(kind == SingleLinkageTree.UpdateKind.UNCHANGED)
                    .as("reported change at step " + step)
                    .isEqualTo(before.equals(after));

            if (kind == SingleLinkageTree.UpdateKind.HEIGHT_ONLY) {
                // the fast path claims only one node height moved, which the operator relies on to
                // avoid installing the whole topology again
                assertThat(structureOf(after))
                        .as("height-only step " + step + " must not move any clade")
                        .isEqualTo(structureOf(before));

                int moved = 0;
                for (int merge = 0; merge < after.size(); merge++) {
                    if (before.get(merge).height() != after.get(merge).height()) {
                        moved++;
                    }
                }
                assertThat(moved).as("heights moved at step " + step).isEqualTo(1);
                assertThat(after.get(clustering.getLastChangedMerge()).height())
                        .as("reported merge at step " + step)
                        .isNotEqualTo(before.get(clustering.getLastChangedMerge()).height());
            }

            clustering.accept();
        }
    }

    @ParameterizedTest
    @ValueSource(longs = {31L, 32L, 33L})
    void restoreUndoesAnUpdate(long seed) {
        Random random = new Random(seed);
        int leafCount = 6;

        double[] distances = randomDistances(leafCount, random);
        SingleLinkageTree clustering = new SingleLinkageTree(leafCount, distances);

        for (int step = 0; step < 500; step++) {
            List<Merge> before = mergesOf(clustering);
            double[] copheneticBefore = copheneticMatrix(clustering);

            int first = random.nextInt(leafCount);
            int second = (first + 1 + random.nextInt(leafCount - 1)) % leafCount;

            clustering.storeDistance(first, second);
            clustering.update(first, second, Math.abs(random.nextGaussian()) + 1e-6);
            clustering.restore();

            assertThat(mergesOf(clustering)).as("step " + step).isEqualTo(before);
            assertThat(copheneticMatrix(clustering)).isEqualTo(copheneticBefore);
        }
    }

    @Test
    void rejectsTooFewTaxa() {
        assertThatThrownBy(() -> new SingleLinkageTree(2, new double[]{1.0}))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("at least three taxa");
    }

    @Test
    void crossPairCountsPartitionAllPairs() {
        Random random = new Random(99L);

        for (int leafCount = 3; leafCount <= 9; leafCount++) {
            SingleLinkageTree clustering =
                    new SingleLinkageTree(leafCount, randomDistances(leafCount, random));

            int total = 0;
            for (int merge = 0; merge < leafCount - 1; merge++) {
                total += clustering.getCrossPairCount(merge);
            }

            assertThat(total).isEqualTo(leafCount * (leafCount - 1) / 2);
        }
    }

    private static double[] randomDistances(int leafCount, Random random) {
        double[] distances = new double[leafCount * (leafCount - 1) / 2];
        for (int pair = 0; pair < distances.length; pair++) {
            distances[pair] = 0.1 + random.nextDouble();
        }
        return distances;
    }

    /** Clades and heights of the clustering, in ascending height order. */
    private static List<Merge> mergesOf(SingleLinkageTree clustering) {
        int leafCount = clustering.getLeafCount();
        BitSet[] clades = new BitSet[2 * leafCount - 1];

        for (int leaf = 0; leaf < leafCount; leaf++) {
            clades[leaf] = new BitSet();
            clades[leaf].set(leaf);
        }

        List<Merge> merges = new ArrayList<>();
        for (int merge = 0; merge < leafCount - 1; merge++) {
            BitSet left = clades[clustering.getMergeLeft(merge)];
            BitSet right = clades[clustering.getMergeRight(merge)];

            BitSet clade = (BitSet) left.clone();
            clade.or(right);
            clades[leafCount + merge] = clade;

            merges.add(new Merge(clade, smaller(left, right), clustering.getMergeHeight(merge)));
        }

        return merges;
    }

    /**
     * Single linkage by the book: walk the pairs in ascending order and merge whenever the two
     * endpoints are not yet connected.
     */
    private static List<Merge> bruteForceMerges(int leafCount, double[] distances) {
        record Pair(int first, int second, double distance) {
        }

        List<Pair> pairs = new ArrayList<>();
        int pair = 0;
        for (int first = 0; first < leafCount; first++) {
            for (int second = first + 1; second < leafCount; second++) {
                pairs.add(new Pair(first, second, distances[pair++]));
            }
        }
        pairs.sort(Comparator.comparingDouble(Pair::distance));

        BitSet[] clusterOf = new BitSet[leafCount];
        for (int leaf = 0; leaf < leafCount; leaf++) {
            clusterOf[leaf] = new BitSet();
            clusterOf[leaf].set(leaf);
        }

        List<Merge> merges = new ArrayList<>();
        for (Pair candidate : pairs) {
            BitSet left = clusterOf[candidate.first()];
            BitSet right = clusterOf[candidate.second()];
            if (left == right) {
                continue;
            }

            BitSet clade = (BitSet) left.clone();
            clade.or(right);
            merges.add(new Merge(clade, smaller(left, right), candidate.distance() / 2.0));

            for (int leaf = clade.nextSetBit(0); leaf >= 0; leaf = clade.nextSetBit(leaf + 1)) {
                clusterOf[leaf] = clade;
            }
        }

        return merges;
    }

    /** The clades of a clustering, with the heights dropped. */
    private static List<BitSet> structureOf(List<Merge> merges) {
        List<BitSet> structure = new ArrayList<>();
        for (Merge merge : merges) {
            structure.add(merge.clade());
            structure.add(merge.smallerChild());
        }
        return structure;
    }

    private static BitSet smaller(BitSet left, BitSet right) {
        if (left.cardinality() != right.cardinality()) {
            return left.cardinality() < right.cardinality() ? left : right;
        }
        return left.nextSetBit(0) < right.nextSetBit(0) ? left : right;
    }

    /** Pairwise distances of the induced tree, that is twice the height of each pair's ancestor. */
    private static double[] copheneticMatrix(SingleLinkageTree clustering) {
        int leafCount = clustering.getLeafCount();
        double[] cophenetic = new double[leafCount * (leafCount - 1) / 2];

        BitSet[] clades = new BitSet[2 * leafCount - 1];
        for (int leaf = 0; leaf < leafCount; leaf++) {
            clades[leaf] = new BitSet();
            clades[leaf].set(leaf);
        }

        for (int merge = 0; merge < leafCount - 1; merge++) {
            BitSet left = clades[clustering.getMergeLeft(merge)];
            BitSet right = clades[clustering.getMergeRight(merge)];

            for (int a = left.nextSetBit(0); a >= 0; a = left.nextSetBit(a + 1)) {
                for (int b = right.nextSetBit(0); b >= 0; b = right.nextSetBit(b + 1)) {
                    cophenetic[clustering.pairIndex(a, b)] = 2.0 * clustering.getMergeHeight(merge);
                }
            }

            BitSet clade = (BitSet) left.clone();
            clade.or(right);
            clades[leafCount + merge] = clade;
        }

        return cophenetic;
    }
}
