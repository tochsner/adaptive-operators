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
 * <p>Everything is indexed by tree node number, and distances are twice the node heights,
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

    /** The splits of every merge but the last, root first; the last merge has nothing to split against. */
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

    /** Walks the merges backwards from the root, recording which clusters were active at each one. */
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
     * {@code S(V, A)}: the mean data-implied distance between two clades. Substitution distances are
     * divided by the rate and extended to the present by both sampling times, which makes them
     * comparable to the merge distances.
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
     * Mean of {@code p_r(V, A)}: A's size-weighted share of the excess of {@code S(V, .)} over the
     * baseline {@code S(A, B)}, so a common scale factor on the scores cancels. The baseline drops
     * to {@code (1 - EXCESS_FLOOR)} times the smaller score when the data places V inside the split.
     * Falls back to {@code |A| / |X|} when both excesses vanish, and is clamped away from 0 and 1.
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
