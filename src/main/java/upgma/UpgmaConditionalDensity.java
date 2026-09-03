package upgma;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.spec.type.RealScalar;
import beast.base.util.GammaFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Evaluates the conditional log density {@code log p(D | T)} of a distance matrix D that is already
 * known to satisfy {@code UPGMA(D) = T}, following the same backward construction as
 * {@link UpgmaDistanceSampler} but reading every quantity directly off D instead of sampling it.
 *
 * <p>For every merge r and every cluster V still active at that point, the fraction {@code p_r(V,
 * A(r))} that the sampler would have drawn is recovered from D via
 * {@code p_r(V, A) = |A| (d(V, A) - d(r)) / (|X| (d(V, X) - d(r)))} &mdash; using {@code s_{r+1}(V,
 * X) + delta_r = d(V, X) - d(r)}, from approach.md's slack recursion. Its log density under the same
 * {@code Beta(alpha, beta)} shape the sampler would have used is accumulated, together with the
 * matching Jacobian factor {@code |X|^2 (d(V, X) - d(r)) / (|A| |B|)} from approach.md's density
 * formula.
 *
 * <p>Cluster-pair distances are read directly off D by averaging over the relevant leaf pairs,
 * rather than propagated recursively through slacks, since the full matrix is already available.
 * The Beta-shape formula is kept textually identical to {@link UpgmaDistanceSampler}'s, since the
 * Hastings ratio in {@code UpgmaRefreshOperator} depends on the two agreeing exactly.
 */
public class UpgmaConditionalDensity {

    // must match UpgmaDistanceSampler.WEIGHT_PSEUDOCOUNT exactly: both add it to weightedA/weightedB
    // before it sets the Beta shape, so a 0/0 total or a 0 shape parameter (from two clusters with
    // identical, or coincidentally tied, substitution distances to V) never arises. Negligible
    // against any real JC69 signal.
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

    public UpgmaConditionalDensity(
            Tree tree, double[][] substitutionDistances, RealScalar<?> substitutionRate, double concentration) {
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
     * Returns {@code log p(D | T)}: the sum of the Beta log densities of every fraction D implies,
     * minus the log Jacobian of the map from those fractions to D.
     */
    public double logDensity(double[][] distances) {
        int mergeCount = this.mergeOrder.length; // n - 1
        if (mergeCount == 0) {
            return 0.0;
        }

        Node root = this.mergeOrder[mergeCount - 1];
        int rootLeft = root.getChild(0).getNr();
        int rootRight = root.getChild(1).getNr();

        // active clusters right after every merge but the last, i.e. P(n - 2): the root's children
        List<Integer> active = new ArrayList<>();
        active.add(rootLeft);
        active.add(rootRight);

        double logDensity = 0.0;

        for (int r = mergeCount - 1; r >= 1; r--) {
            Node x = this.mergeOrder[r - 1];
            int aId = x.getChild(0).getNr();
            int bId = x.getChild(1).getNr();
            int aSize = this.clade[aId].length;
            int bSize = this.clade[bId].length;
            int xSize = aSize + bSize;
            double dr = this.mergeDistance[r];

            active.remove(Integer.valueOf(x.getNr()));

            for (int v : active) {
                int[] vLeaves = this.clade[v];

                double dVA = clusterDistance(distances, vLeaves, this.clade[aId]);
                double dVB = clusterDistance(distances, vLeaves, this.clade[bId]);
                double dVX = (aSize * dVA + bSize * dVB) / xSize;

                double base = xSize * (dVX - dr);
                double p = aSize * (dVA - dr) / base;

                double weightedA = aSize * substitutionScore(vLeaves, this.clade[aId]) + WEIGHT_PSEUDOCOUNT;
                double weightedB = bSize * substitutionScore(vLeaves, this.clade[bId]) + WEIGHT_PSEUDOCOUNT;
                double total = weightedA + weightedB;
                double alpha = this.concentration * weightedA / total;
                double beta = this.concentration * weightedB / total;

                logDensity += logBetaPdf(p, alpha, beta);
                logDensity -= Math.log(xSize) + Math.log(base) - Math.log(aSize) - Math.log(bSize);
            }

            active.add(aId);
            active.add(bId);
        }

        return logDensity;
    }

    /** Average distance in D between two disjoint leaf sets, i.e. the average-linkage cluster distance. */
    private double clusterDistance(double[][] distances, int[] leavesV, int[] leavesA) {
        double sum = 0.0;
        for (int v : leavesV) {
            for (int a : leavesA) {
                sum += distances[v][a];
            }
        }
        return sum / (leavesV.length * (double) leavesA.length);
    }

    /** {@code S_r(V, A)}: the mean substitution distance between two disjoint leaf sets. */
    private double substitutionScore(int[] leavesV, int[] leavesA) {
        double sum = 0.0;
        for (int v : leavesV) {
            for (int a : leavesA) {
                sum += this.substitutionDistances[v][a] / this.substitutionRate.get();
            }
        }
        return sum / (leavesV.length * (double) leavesA.length);
    }

    private static double logBetaPdf(double p, double alpha, double beta) {
        double logNormalizer =
                GammaFunction.lnGamma(alpha) + GammaFunction.lnGamma(beta) - GammaFunction.lnGamma(alpha + beta);
        return (alpha - 1.0) * Math.log(p) + (beta - 1.0) * Math.log(1.0 - p) - logNormalizer;
    }
}
