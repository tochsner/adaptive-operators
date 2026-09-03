package upgma;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.spec.type.RealScalar;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Draws a distance matrix D that is consistent with a fixed tree, following the backward
 * slack-sampling construction in {@code approach.md}.
 *
 * <p>Average-linkage clustering repeatedly merges the closest pair of clusters. This class runs
 * that process in reverse: starting from the root, it repeatedly splits the most recently formed
 * cluster back into the two children it was built from, and for every other cluster still active at
 * that point draws how much of the accumulated slack against the split cluster falls to each of the
 * two new clusters. Every pairwise slack between individual taxa is fully determined once every
 * internal node has been split back down to its leaves, at which point the distance matrix follows
 * directly from {@code d(i, j) = d(1) + s_1(i, j)}.
 *
 * <p>Everything here is indexed by tree leaf node number (0..n-1), matching the convention used by
 * {@code adapters.TreeUtils} and {@code distancematrix.SingleLinkageTree} elsewhere in this
 * codebase, so the resulting matrix can be handed straight to {@code TreeUtils.applyClustering}
 * without remapping.
 *
 * <p>Distances in D are in the same units as the tree's node heights, doubled: {@code d(r) = 2 *
 * height(r)}, the standard cophenetic convention (see
 * {@code distancematrix.SingleLinkageTree#writeCopheneticMatrix}), so that the distance between two
 * contemporaneous tips equals twice the height of their most recent common ancestor.
 */
public class UpgmaDistanceSampler {

    // the Beta mean is clamped to [MIN_MEAN, 1 - MIN_MEAN] so neither shape parameter can collapse to
    // 0 when the data says V is (almost) exactly as far from A as A is from B. Must match between
    // UpgmaDistanceSampler and UpgmaConditionalDensity, since the Hastings ratio depends on both
    // using the same Beta shapes.
    private static final double MIN_MEAN = 0.1;

    private final int n;
    private final double concentration;
    private final double[][] substitutionDistances;
    private final double[] tipDates;
    private final RealScalar<?> substitutionRate;

    /** Internal nodes in ascending height order: {@code mergeOrder[r - 1]} is the tree's r-th merge. */
    private final Node[] mergeOrder;

    /** {@code d(r) = 2 * height} of the r-th merge, 1-indexed; index 0 is unused. */
    private final double[] mergeDistance;

    /** Leaf indices under every node, indexed by node number. */
    private final int[][] clade;

    public UpgmaDistanceSampler(Tree tree, double[][] substitutionDistances, double[] tipDates, RealScalar<?> substitutionRate, double concentration) {
        this.n = tree.getLeafNodeCount();
        this.substitutionDistances = substitutionDistances;
        this.tipDates = tipDates;
        this.substitutionRate = substitutionRate;
        this.concentration = concentration;

        this.mergeOrder = internalNodesByAscendingHeight(tree);
        this.mergeDistance = new double[this.mergeOrder.length + 1];
        for (int r = 1; r <= this.mergeOrder.length; r++) {
            this.mergeDistance[r] = 2.0 * this.mergeOrder[r - 1].getHeight();
        }

        this.clade = new int[2 * this.n - 1][];
        computeClades(tree.getRoot());
    }

    private static Node[] internalNodesByAscendingHeight(Tree tree) {
        int leafCount = tree.getLeafNodeCount();
        Node[] internalNodes = new Node[leafCount - 1];
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
        if (node.isLeaf()) {
            int[] leaves = {node.getNr()};
            this.clade[node.getNr()] = leaves;
            return leaves;
        }

        int[] left = computeClades(node.getChild(0));
        int[] right = computeClades(node.getChild(1));
        int[] leaves = new int[left.length + right.length];
        System.arraycopy(left, 0, leaves, 0, left.length);
        System.arraycopy(right, 0, leaves, left.length, right.length);
        this.clade[node.getNr()] = leaves;
        return leaves;
    }

    /**
     * Draws slacks backwards from the root and writes the resulting distance matrix into
     * {@code distances}, indexed by tree leaf node number.
     */
    public void sample(double[][] distances) {
        int mergeCount = this.mergeOrder.length; // n - 1

        for (int i = 0; i < this.n; i++) {
            distances[i][i] = 0.0;
        }

        if (mergeCount == 0) {
            return;
        }

        // slack[a][b], for a < b, holds s_{level+1}(a, b) for the clusters active at the current
        // backward step, indexed by tree node number
        double[][] slack = new double[2 * this.n - 1][2 * this.n - 1];

        Node root = this.mergeOrder[mergeCount - 1];
        int rootLeft = root.getChild(0).getNr();
        int rootRight = root.getChild(1).getNr();

        // active clusters right after every merge but the last, i.e. P(n - 2): the root's children.
        // s_{n-1}(rootLeft, rootRight) = 0, already the default value in `slack`.
        List<Integer> active = new ArrayList<>();
        active.add(rootLeft);
        active.add(rootRight);

        for (int r = mergeCount - 1; r >= 1; r--) {
            Node x = this.mergeOrder[r - 1];
            int xId = x.getNr();
            int aId = x.getChild(0).getNr();
            int bId = x.getChild(1).getNr();
            int aSize = this.clade[aId].length;
            int bSize = this.clade[bId].length;
            int xSize = aSize + bSize;

            double dr = this.mergeDistance[r];
            double delta = this.mergeDistance[r + 1] - dr;

            active.remove(Integer.valueOf(xId));

            // unaffected pairs: s_r(U, V) = s_{r+1}(U, V) + delta
            for (int i = 0; i < active.size(); i++) {
                for (int j = i + 1; j < active.size(); j++) {
                    int u = active.get(i);
                    int v = active.get(j);
                    setSlack(slack, u, v, getSlack(slack, u, v) + delta);
                }
            }

            // affected pairs: split every other active cluster's slack against X between A and B
            for (int v : active) {
                double vx = getSlack(slack, v, xId);
                double base = xSize * (vx + delta);

                double mean = betaMean(
                        substitutionScore(this.clade[v], this.clade[aId]) - dr,
                        substitutionScore(this.clade[v], this.clade[bId]) - dr,
                        aSize, bSize);

                double p = sampleBeta(this.concentration * mean, this.concentration * (1.0 - mean));
                setSlack(slack, v, aId, p * base / aSize);
                setSlack(slack, v, bId, (1.0 - p) * base / bSize);
            }

            setSlack(slack, aId, bId, 0.0);

            active.add(aId);
            active.add(bId);
        }

        double base = this.mergeDistance[1];
        for (int i = 0; i < this.n; i++) {
            for (int j = i + 1; j < this.n; j++) {
                double d = base + getSlack(slack, i, j);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }
    }

    /**
     * {@code S_r(V, A)}: the mean data-implied distance between two disjoint leaf sets, in the units of
     * D. Each pairwise substitution distance is converted to time by the substitution rate and
     * extended to the present by adding both sampling times, so that it is directly comparable to
     * the merge distances {@code d(r) = 2 * height(r)}.
     */
    private double substitutionScore(int[] leavesV, int[] leavesA) {
        double rate = this.substitutionRate.get();
        double sum = 0.0;
        for (int v : leavesV) {
            for (int a : leavesA) {
                sum += this.substitutionDistances[v][a] / rate + this.tipDates[v] + this.tipDates[a];
            }
        }
        return sum / (leavesV.length * leavesA.length);
    }

    /** Draws Beta(alpha, beta) via two Gamma draws, since {@code Randomizer} has no direct Beta sampler. */
    private static double sampleBeta(double alpha, double beta) {
        double x = Randomizer.nextGamma(alpha, 1.0);
        double y = Randomizer.nextGamma(beta, 1.0);
        return x / (x + y);
    }


    /**
     * Mean of {@code p_r(V, A)}: the fraction of the excess {@code |A|(S(V, A) - d(r))} over the total
     * excess {@code |A|(S(V, A) - d(r)) + |B|(S(V, B) - d(r))}, with negative excesses clipped to 0.
     * Falls back to the no-information value {@code |A| / |X|} when the data places V no further
     * from either child than the merge distance, and is clamped away from 0 and 1.
     */
    private static double betaMean(double excessA, double excessB, int aSize, int bSize) {
        double weightedA = aSize * Math.max(excessA, 0.0);
        double weightedB = bSize * Math.max(excessB, 0.0);
        double total = weightedA + weightedB;

        double mean = total > 0.0 ? weightedA / total : (double) aSize / (aSize + bSize);
        return Math.min(Math.max(mean, MIN_MEAN), 1.0 - MIN_MEAN);
    }

    private static double getSlack(double[][] slack, int a, int b) {
        return a < b ? slack[a][b] : slack[b][a];
    }

    private static void setSlack(double[][] slack, int a, int b, double value) {
        if (a < b) {
            slack[a][b] = value;
        } else {
            slack[b][a] = value;
        }
    }
}
