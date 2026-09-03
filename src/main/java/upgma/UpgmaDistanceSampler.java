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

    // added to weightedA/weightedB before they set the Beta shape, so that two clusters with
    // identical (or coincidentally tied) substitution distances to V never produce a 0/0 total or
    // a 0 shape parameter (Gamma(0, .) is degenerate). Negligible against any real JC69 signal.
    private static final double WEIGHT_PSEUDOCOUNT = 1e-8;

    private final int n;
    private final double concentration;
    private final double[][] substitutionDistances;
    private final RealScalar<?> substitutionRate;

    /** Internal nodes in ascending height order: {@code mergeOrder[r - 1]} is the tree's r-th merge. */
    private final Node[] mergeOrder;

    /** {@code d(r) = 2 * height} of the r-th merge, 1-indexed; index 0 is unused. */
    private final double[] mergeDistance;

    /** Leaf indices under every node, indexed by node number. */
    private final int[][] clade;

    public UpgmaDistanceSampler(Tree tree, double[][] substitutionDistances, RealScalar<?> substitutionRate, double concentration) {
        this.n = tree.getLeafNodeCount();
        this.substitutionDistances = substitutionDistances;
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

            double delta = this.mergeDistance[r + 1] - this.mergeDistance[r];

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

                double weightedA = aSize * substitutionScore(this.clade[v], this.clade[aId]) + WEIGHT_PSEUDOCOUNT;
                double weightedB = bSize * substitutionScore(this.clade[v], this.clade[bId]) + WEIGHT_PSEUDOCOUNT;
                double total = weightedA + weightedB;

                double p = sampleBeta(this.concentration * weightedA / total, this.concentration * weightedB / total);
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

    /** {@code S_r(V, A)}: the mean substitution distance between two disjoint leaf sets. */
    private double substitutionScore(int[] leavesV, int[] leavesA) {
        double sum = 0.0;
        for (int v : leavesV) {
            for (int a : leavesA) {
                sum += this.substitutionDistances[v][a] / this.substitutionRate.get();
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
