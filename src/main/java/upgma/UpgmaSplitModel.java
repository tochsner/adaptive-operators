package upgma;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.spec.type.RealScalar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * The tree- and data-derived quantities that {@link UpgmaDistanceSampler} and
 * {@link UpgmaConditionalDensity} share: the backward walk over the tree's merges, and the Beta
 * distribution every slack fraction is drawn from and scored under.
 *
 * <p>Average-linkage clustering repeatedly merges the closest pair of clusters. Both classes run
 * that process in reverse, splitting the most recently formed cluster back into the two children it
 * was built from; {@link #splits()} enumerates those splits, from the root downwards, together with
 * the clusters that were active at the time. The last merge is not a split: the root's two children
 * have no third cluster to divide slack against.
 *
 * <p>Everything is indexed by tree node number, and distances are in the same units as the node
 * heights, doubled: {@code d(r) = 2 * height(r)}, the standard cophenetic convention (see
 * {@code distancematrix.SingleLinkageTree#writeCopheneticMatrix}), so that the distance between two
 * contemporaneous tips equals twice the height of their most recent common ancestor.
 *
 * <p>A model is an immutable snapshot of the tree and the substitution rate it was built from, so
 * the operator rebuilds it whenever either changes.
 */
public class UpgmaSplitModel {

    // the Beta mean is clamped to [MIN_MEAN, 1 - MIN_MEAN] so neither shape parameter can collapse to
    // 0 when the data says V is (almost) exactly as far from A as A is from B
    private static final double MIN_MEAN = 0.1;

    // the baseline the excesses are measured from is capped at (1 - EXCESS_FLOOR) times the smaller of
    // V's two scores, so the closer child always keeps a relative excess of at least EXCESS_FLOOR and
    // the difference between the two scores stays visible even when the data places V inside the split
    private static final double EXCESS_FLOOR = 0.1;

    /**
     * One backward step: the {@code r}-th merge X, split back into its children A and B.
     * {@code active} holds the other clusters present just after merge {@code r}, i.e. those whose
     * slack against X has to be divided between A and B.
     */
    public record Split(int r, int nodeNr, int leftNr, int rightNr, int leftSize, int rightSize, int[] active) {
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

    /** Leaf indices under every node, indexed by node number. */
    private final int[][] clade;

    private final List<Split> splits;

    /** {@code S(A, B)} of the split at a node, indexed by node number; NaN until first needed. */
    private final double[] splitScore;

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

        this.clade = new int[2 * this.n - 1][];
        this.computeClades(tree.getRoot());

        this.splits = this.collectSplits(mergeOrder);

        this.splitScore = new double[2 * this.n - 1];
        Arrays.fill(this.splitScore, Double.NaN);
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

    /** The splits of every merge but the last, in backward order: the root's children first. */
    public List<Split> splits() {
        return this.splits;
    }

    public int[] clade(int nodeNr) {
        return this.clade[nodeNr];
    }

    public int cladeSize(int nodeNr) {
        return this.clade[nodeNr].length;
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

    private int[] computeClades(Node node) {
        int[] leaves;
        if (node.isLeaf()) {
            leaves = new int[]{node.getNr()};
        } else {
            int[] left = this.computeClades(node.getChild(0));
            int[] right = this.computeClades(node.getChild(1));
            leaves = new int[left.length + right.length];
            System.arraycopy(left, 0, leaves, 0, left.length);
            System.arraycopy(right, 0, leaves, left.length, right.length);
        }

        this.clade[node.getNr()] = leaves;
        return leaves;
    }

    /**
     * Walks the merges backwards from the root, recording for each one which clusters were active
     * when it happened. The walk depends on the topology alone, so it is done once per tree.
     */
    private List<Split> collectSplits(Node[] mergeOrder) {
        int mergeCount = mergeOrder.length;
        if (mergeCount == 0) {
            return List.of();
        }

        // the clusters right after every merge but the last, i.e. P(n - 2): the root's children
        Node root = mergeOrder[mergeCount - 1];
        List<Integer> active = new ArrayList<>();
        active.add(root.getChild(0).getNr());
        active.add(root.getChild(1).getNr());

        List<Split> splits = new ArrayList<>(mergeCount - 1);
        for (int r = mergeCount - 1; r >= 1; r--) {
            Node x = mergeOrder[r - 1];
            int leftNr = x.getChild(0).getNr();
            int rightNr = x.getChild(1).getNr();

            active.remove(Integer.valueOf(x.getNr()));
            splits.add(new Split(
                    r, x.getNr(), leftNr, rightNr, this.cladeSize(leftNr), this.cladeSize(rightNr),
                    active.stream().mapToInt(Integer::intValue).toArray()));

            active.add(leftNr);
            active.add(rightNr);
        }

        return splits;
    }

    /**
     * {@code S_r(V, A)}: the mean data-implied distance between two clades, in the units of D. Each
     * pairwise substitution distance is converted to time by the substitution rate and extended to
     * the present by adding both sampling times, so that it is directly comparable to the merge
     * distances {@code d(r) = 2 * height(r)}.
     */
    public double substitutionScore(int nodeV, int nodeA) {
        double sum = 0.0;
        for (int v : this.clade[nodeV]) {
            for (int a : this.clade[nodeA]) {
                sum += this.substitutionDistances[v][a] / this.substitutionRate
                        + this.tipDates[v] + this.tipDates[a];
            }
        }
        return sum / (this.cladeSize(nodeV) * (double) this.cladeSize(nodeA));
    }

    /** {@code S(A, B)}: the data-implied merge distance of a split, computed once per split. */
    private double splitScore(Split split) {
        if (Double.isNaN(this.splitScore[split.nodeNr()])) {
            this.splitScore[split.nodeNr()] = this.substitutionScore(split.leftNr(), split.rightNr());
        }
        return this.splitScore[split.nodeNr()];
    }

    /**
     * Mean of {@code p_r(V, A)}: the fraction of the excess {@code |A|(S(V, A) - b)} over the total
     * excess {@code |A|(S(V, A) - b) + |B|(S(V, B) - b)}. The baseline {@code b} is the data's own
     * merge distance of the split {@code S(A, B)}, so the mean is a function of topology and data
     * alone and a common scale factor on all scores (a miscalibrated rate) cancels. When the data
     * places V inside the split, i.e. closer to a child than the children are to each other, the
     * baseline is capped at {@code (1 - EXCESS_FLOOR)} times the smaller score instead, which keeps
     * the difference between the two scores as a graded signal rather than clipping it away. Falls
     * back to the no-information value {@code |A| / |X|} only when both scores are 0, and is clamped
     * away from 0 and 1.
     */
    public double betaMean(int nodeV, Split split) {
        double splitScore = this.splitScore(split);
        double scoreVA = this.substitutionScore(nodeV, split.leftNr());
        double scoreVB = this.substitutionScore(nodeV, split.rightNr());

        double baseline = Math.min(splitScore, (1.0 - EXCESS_FLOOR) * Math.min(scoreVA, scoreVB));
        double weightedA = split.leftSize() * Math.max(scoreVA - baseline, 0.0);
        double weightedB = split.rightSize() * Math.max(scoreVB - baseline, 0.0);
        double total = weightedA + weightedB;

        double mean = total > 0.0 ? weightedA / total : split.leftSize() / (double) split.size();
        return Math.min(Math.max(mean, MIN_MEAN), 1.0 - MIN_MEAN);
    }
}
