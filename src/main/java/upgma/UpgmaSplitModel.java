package upgma;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.spec.type.RealScalar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * The quantities {@link UpgmaDistanceSampler} and {@link UpgmaConditionalDensity} share: the tree's
 * merges walked backwards from the root, and the Beta distribution every slack fraction is drawn
 * from and scored under.
 *
 * <p>The Beta means depend on the tree and the data alone, never on the distance matrix being drawn
 * or scored, so all of them are worked out up front in {@link #computeBetaMeans} and read back by
 * offset. Everything is indexed by tree node number, and distances are twice the node heights,
 * {@code d(r) = 2 * height(r)}, the usual cophenetic convention. A model is an immutable snapshot
 * of the tree and substitution rate it was built from.
 */
public class UpgmaSplitModel {

    /** Keeps the Beta mean away from 0 and 1, so neither shape parameter can collapse. */
    public static final double MIN_MEAN = 0.1;

    // the smallest relative excess the closer child keeps, so the two scores stay distinguishable
    // even when the data places V inside the split
    private static final double EXCESS_FLOOR = 0.1;

    /**
     * One backward step: the {@code r}-th merge X split back into A and B, against the clusters
     * {@code active} at that point, whose slack against X has to be divided between the two.
     * {@code meanOffset} is where this split's Beta means start, one per entry of {@code active}.
     */
    public record Split(
            int r, int nodeNr, int leftNr, int rightNr, int leftSize, int rightSize, int[] active,
            int meanOffset) {
        public int size() {
            return this.leftSize + this.rightSize;
        }
    }

    private final int n;
    private final double concentration;

    /** {@code d(r) = 2 * height} of the r-th merge, 1-indexed; index 0 is unused. */
    private final double[] mergeDistance;

    /** Number of leaves under every node, indexed by node number. */
    private final int[] cladeSize;

    private final List<Split> splits;

    /** The clusters the backward walk starts from: the root's two children. */
    private final int[] initialClusters;

    /** Every split's Beta means, laid out back to back in {@link Split#meanOffset} order. */
    private final double[] betaMean;

    public UpgmaSplitModel(
            Tree tree, double[][] substitutionDistances, double[] tipDates, RealScalar<?> substitutionRate,
            double concentration) {
        this(tree, leafScores(substitutionDistances, tipDates, substitutionRate.get()), concentration,
                new double[tree.getLeafNodeCount() * tree.getLeafNodeCount()]);
    }

    /**
     * @param leafScores flat {@code n x n} matrix of {@link #leafScores}, indexed by leaf node number
     * @param scratch    working memory of at least {@code n * n} doubles, only used during construction
     */
    public UpgmaSplitModel(Tree tree, double[] leafScores, double concentration, double[] scratch) {
        this.n = tree.getLeafNodeCount();
        this.concentration = concentration;

        Node[] mergeOrder = internalNodesByAscendingHeight(tree);
        this.mergeDistance = new double[mergeOrder.length + 1];
        for (int r = 1; r <= mergeOrder.length; r++) {
            this.mergeDistance[r] = 2.0 * mergeOrder[r - 1].getHeight();
        }

        this.cladeSize = new int[2 * this.n - 1];
        computeCladeSizes(tree, this.cladeSize);

        if (mergeOrder.length == 0) {
            this.initialClusters = new int[0];
        } else {
            Node root = mergeOrder[mergeOrder.length - 1];
            this.initialClusters = new int[]{root.getChild(0).getNr(), root.getChild(1).getNr()};
        }

        this.splits = this.collectSplits(mergeOrder);

        int meanCount = this.splits.isEmpty() ? 0
                : this.splits.getLast().meanOffset() + this.splits.getLast().active().length;
        this.betaMean = new double[meanCount];
        this.computeBetaMeans(leafScores, scratch);
    }

    /**
     * The leaf-level scores {@code S(i, j) = d_sub(i, j) / rate + tipDate(i) + tipDate(j)} the Beta
     * means are built from, as a flat row-major {@code n x n} matrix with a zero diagonal. They only
     * change with the substitution rate, so a caller can keep them across models.
     */
    public static double[] leafScores(double[][] substitutionDistances, double[] tipDates, double substitutionRate) {
        int n = tipDates.length;
        double[] scores = new double[n * n];
        for (int i = 0; i < n; i++) {
            double[] row = substitutionDistances[i];
            double tipDate = tipDates[i];
            for (int j = i + 1; j < n; j++) {
                double score = row[j] / substitutionRate + tipDate + tipDates[j];
                scores[i * n + j] = score;
                scores[j * n + i] = score;
            }
        }
        // the diagonal is never a cluster pair, so it stays out of the averages
        return scores;
    }

    public int leafCount() {
        return this.n;
    }

    public int mergeCount() {
        return this.n - 1;
    }

    public double concentration() {
        return this.concentration;
    }

    /** {@code d(r) = 2 * height(r)} of the r-th merge, 1-indexed. */
    public double mergeDistance(int r) {
        return this.mergeDistance[r];
    }

    /** The splits of every merge but the last, root first; the last merge has nothing to split against. */
    public List<Split> splits() {
        return this.splits;
    }

    /**
     * The root's two children, {@code P(n - 2)}: the clusters present before the first split, exactly
     * {@code mergeDistance(mergeCount())} apart.
     */
    public int[] initialClusters() {
        return this.initialClusters;
    }

    public int cladeSize(int nodeNr) {
        return this.cladeSize[nodeNr];
    }

    /**
     * Mean of {@code p_r(V, A)} for the {@code activeIndex}-th cluster of a split: A's size-weighted
     * share of the excess of {@code S(V, .)} over the baseline {@code S(A, B)}, so a common scale
     * factor on the scores cancels. The baseline drops to {@code (1 - EXCESS_FLOOR)} times the
     * smaller score when the data places V inside the split. Falls back to {@code |A| / |X|} when
     * both excesses vanish, and is clamped away from 0 and 1.
     */
    public double betaMean(Split split, int activeIndex) {
        return this.betaMean[split.meanOffset() + activeIndex];
    }

    public double alpha(double mean) {
        return this.concentration * mean;
    }

    public double beta(double mean) {
        return this.concentration * (1.0 - mean);
    }

    private static Node[] internalNodesByAscendingHeight(Tree tree) {
        Node[] internalNodes = new Node[tree.getLeafNodeCount() - 1];
        int k = 0;
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                internalNodes[k++] = node;
            }
        }
        Arrays.sort(internalNodes, Comparator.comparingDouble(Node::getHeight));
        return internalNodes;
    }

    /**
     * Counts the leaves under every node, deepest first. The walk is iterative rather than
     * recursive, since a tree of a few thousand taxa can be deep enough to overflow the stack.
     */
    private static void computeCladeSizes(Tree tree, int[] cladeSize) {
        Node[] postOrder = new Node[tree.getNodeCount()];
        int count = 0;

        Node[] pending = new Node[tree.getNodeCount()];
        int pendingCount = 0;
        pending[pendingCount++] = tree.getRoot();
        while (pendingCount > 0) {
            Node node = pending[--pendingCount];
            postOrder[count++] = node;
            for (int c = 0; c < node.getChildCount(); c++) {
                pending[pendingCount++] = node.getChild(c);
            }
        }

        for (int i = count - 1; i >= 0; i--) {
            Node node = postOrder[i];
            if (node.isLeaf()) {
                cladeSize[node.getNr()] = 1;
            } else {
                int size = 0;
                for (int c = 0; c < node.getChildCount(); c++) {
                    size += cladeSize[node.getChild(c).getNr()];
                }
                cladeSize[node.getNr()] = size;
            }
        }
    }

    /** Walks the merges backwards from the root, recording which clusters were active at each one. */
    private List<Split> collectSplits(Node[] mergeOrder) {
        int mergeCount = mergeOrder.length;
        if (mergeCount == 0) {
            return List.of();
        }

        // the clusters right after every merge but the last, i.e. P(n - 2): the root's children.
        // the list keeps its insertion order, so removing X shifts everything after it down
        int[] active = new int[this.n];
        active[0] = this.initialClusters[0];
        active[1] = this.initialClusters[1];
        int activeCount = 2;

        List<Split> splits = new ArrayList<>(mergeCount - 1);
        int meanOffset = 0;
        for (int r = mergeCount - 1; r >= 1; r--) {
            Node x = mergeOrder[r - 1];
            int nodeNr = x.getNr();
            int leftNr = x.getChild(0).getNr();
            int rightNr = x.getChild(1).getNr();

            int position = 0;
            while (active[position] != nodeNr) {
                position++;
            }
            System.arraycopy(active, position + 1, active, position, activeCount - position - 1);
            activeCount--;

            splits.add(new Split(
                    r, nodeNr, leftNr, rightNr, this.cladeSize(leftNr), this.cladeSize(rightNr),
                    Arrays.copyOf(active, activeCount), meanOffset));
            meanOffset += activeCount;

            active[activeCount++] = leftNr;
            active[activeCount++] = rightNr;
        }

        return splits;
    }

    /**
     * Fills {@link #betaMean} by walking the merges forwards, the way the clustering itself would.
     *
     * <p>{@code S(V, A)} is an average-linkage cluster distance in its own right, over the matrix
     * {@code d_sub / rate + tipDate + tipDate}, so it obeys the same weighted-average recursion the
     * clustering uses: {@code S(V, A1 + A2) = (|A1| S(V, A1) + |A2| S(V, A2)) / |A|}. Propagating it
     * along the merges therefore produces every score the splits need in one pass over a dense
     * working matrix, rather than averaging over the leaf pairs of each clade pair separately.
     *
     * <p>Just before merge r the working matrix holds {@code S(V, A)} and {@code S(V, B)} for every
     * cluster active at that point, and {@code S(A, B)} is the entry being merged over: the three
     * scores a split's Beta mean is built from. Every leaf starts in its own slot and X takes A's
     * slot; B's slot simply goes dead, since the split lists the clusters still active and nothing
     * ever scans the whole matrix.
     */
    private void computeBetaMeans(double[] leafScores, double[] score) {
        if (this.splits.isEmpty()) {
            return;
        }

        int n = this.n;
        System.arraycopy(leafScores, 0, score, 0, n * n);
        int[] slotOf = new int[2 * n - 1];
        for (int leaf = 0; leaf < n; leaf++) {
            slotOf[leaf] = leaf;
        }

        // the splits run from the root down, so the merges are that list read backwards
        for (int index = this.splits.size() - 1; index >= 0; index--) {
            Split split = this.splits.get(index);
            int slotA = slotOf[split.leftNr()];
            int slotB = slotOf[split.rightNr()];
            int rowA = slotA * n;
            int rowB = slotB * n;
            int leftSize = split.leftSize();
            int rightSize = split.rightSize();
            int size = leftSize + rightSize;

            double splitScore = score[rowA + slotB];
            int[] active = split.active();
            int offset = split.meanOffset();
            for (int i = 0; i < active.length; i++) {
                int slotV = slotOf[active[i]];
                double scoreVA = score[rowA + slotV];
                double scoreVB = score[rowB + slotV];
                this.betaMean[offset + i] = mean(scoreVA, scoreVB, splitScore, leftSize, rightSize);

                // fold B into A's slot, which now stands for X
                double updated = (leftSize * scoreVA + rightSize * scoreVB) / size;
                score[rowA + slotV] = updated;
                score[slotV * n + slotA] = updated;
            }
            slotOf[split.nodeNr()] = slotA;
        }
    }

    private static double mean(
            double scoreVA, double scoreVB, double splitScore, int leftSize, int rightSize) {
        double baseline = Math.min(splitScore, (1.0 - EXCESS_FLOOR) * Math.min(scoreVA, scoreVB));
        double weightedA = leftSize * Math.max(scoreVA - baseline, 0.0);
        double weightedB = rightSize * Math.max(scoreVB - baseline, 0.0);
        double total = weightedA + weightedB;

        double mean = total > 0.0 ? weightedA / total : leftSize / (double) (leftSize + rightSize);
        return Math.min(Math.max(mean, MIN_MEAN), 1.0 - MIN_MEAN);
    }
}
