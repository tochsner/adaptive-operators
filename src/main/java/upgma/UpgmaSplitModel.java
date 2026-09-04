package upgma;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.spec.type.RealScalar;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
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

    // keeps the Beta mean away from 0 and 1, so neither shape parameter can collapse
    private static final double MIN_MEAN = 0.1;

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
    private final double[][] substitutionDistances;
    private final double[] tipDates;
    private final double substitutionRate;

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
        this.n = tree.getLeafNodeCount();
        this.substitutionDistances = substitutionDistances;
        this.tipDates = tipDates;
        this.substitutionRate = substitutionRate.get();
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
        this.computeBetaMeans();
    }

    public int leafCount() {
        return this.n;
    }

    public int mergeCount() {
        return this.n - 1;
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

        Deque<Node> pending = new ArrayDeque<>();
        pending.push(tree.getRoot());
        while (!pending.isEmpty()) {
            Node node = pending.pop();
            postOrder[count++] = node;
            for (Node child : node.getChildren()) {
                pending.push(child);
            }
        }

        for (int i = count - 1; i >= 0; i--) {
            Node node = postOrder[i];
            if (node.isLeaf()) {
                cladeSize[node.getNr()] = 1;
            } else {
                int size = 0;
                for (Node child : node.getChildren()) {
                    size += cladeSize[child.getNr()];
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

        // the clusters right after every merge but the last, i.e. P(n - 2): the root's children
        List<Integer> active = new ArrayList<>();
        active.add(this.initialClusters[0]);
        active.add(this.initialClusters[1]);

        List<Split> splits = new ArrayList<>(mergeCount - 1);
        int meanOffset = 0;
        for (int r = mergeCount - 1; r >= 1; r--) {
            Node x = mergeOrder[r - 1];
            int leftNr = x.getChild(0).getNr();
            int rightNr = x.getChild(1).getNr();

            active.remove(Integer.valueOf(x.getNr()));
            splits.add(new Split(
                    r, x.getNr(), leftNr, rightNr, this.cladeSize(leftNr), this.cladeSize(rightNr),
                    active.stream().mapToInt(Integer::intValue).toArray(), meanOffset));
            meanOffset += active.size();

            active.add(leftNr);
            active.add(rightNr);
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
     * scores a split's Beta mean is built from.
     */
    private void computeBetaMeans() {
        if (this.splits.isEmpty()) {
            return;
        }

        double[] score = new double[this.n * this.n];
        int[] slotOf = new int[2 * this.n - 1];
        int[] clusterOf = new int[this.n];
        for (int leaf = 0; leaf < this.n; leaf++) {
            slotOf[leaf] = leaf;
            clusterOf[leaf] = leaf;
            for (int other = 0; other < this.n; other++) {
                score[leaf * this.n + other] =
                        this.substitutionDistances[leaf][other] / this.substitutionRate
                                + this.tipDates[leaf] + this.tipDates[other];
            }
            // the diagonal is never a cluster pair, so it stays out of the averages
            score[leaf * this.n + leaf] = 0.0;
        }

        // the splits run from the root down, so the merges are that list read backwards
        int activeCount = this.n;
        for (int index = this.splits.size() - 1; index >= 0; index--) {
            Split split = this.splits.get(index);
            int slotA = slotOf[split.leftNr()];
            int slotB = slotOf[split.rightNr()];

            double splitScore = score[slotA * this.n + slotB];
            int[] active = split.active();
            for (int i = 0; i < active.length; i++) {
                int slotV = slotOf[active[i]];
                this.betaMean[split.meanOffset() + i] = mean(
                        score[slotV * this.n + slotA], score[slotV * this.n + slotB], splitScore,
                        split.leftSize(), split.rightSize());
            }

            // fold B into A's slot, which now stands for X
            for (int k = 0; k < activeCount; k++) {
                if (k == slotA || k == slotB) {
                    continue;
                }
                double updated = (split.leftSize() * score[slotA * this.n + k]
                        + split.rightSize() * score[slotB * this.n + k]) / split.size();
                score[slotA * this.n + k] = updated;
                score[k * this.n + slotA] = updated;
            }
            slotOf[split.nodeNr()] = slotA;
            clusterOf[slotA] = split.nodeNr();

            // compact away B's now-stale slot by moving the last active slot into it
            int last = activeCount - 1;
            if (slotB != last) {
                for (int k = 0; k < last; k++) {
                    score[slotB * this.n + k] = score[last * this.n + k];
                    score[k * this.n + slotB] = score[k * this.n + last];
                }
                slotOf[clusterOf[last]] = slotB;
                clusterOf[slotB] = clusterOf[last];
            }
            activeCount--;
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
