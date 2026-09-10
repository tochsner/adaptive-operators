package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Immutable adapter that turns a tree into a fixed-length vector of scalar summary statistics.
 */
public class TreeDescriptionAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");

    private Tree tree;
    private List<TreeDescriptor> descriptors;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();

        this.descriptors = List.of(
                // size and composition
                new TreeHeight(),

                // branch-length structure
                new TreeLength(),
                new MeanBranchLength(),
                new MaxBranchLength(),
                new MinBranchLength(),
                new BranchLengthVariance(),
                new BranchLengthCoefficientOfVariation(),
                new BranchLengthSkewness(),
                new BranchLengthKurtosis(),
                new BranchLengthGini(),
                new BranchLengthEntropy(),
                new MeanExternalBranchLength(),
                new MeanInternalBranchLength(),
                new InternalToExternalBranchLengthRatio(),
                new Treeness(),
                new MeanStemminess(),

                // node-height (time) structure
                new MeanInternalNodeHeight(),
                new InternalNodeHeightVariance(),
                new RootToTipDistanceVariance(),
                new RelativeTimeToFirstSplit(),
                new BasalBranchLengthRatio(),

                // topological balance
                new CollessIndex(),
                new NormalizedCollessIndex(),
                new SackinIndex(),
                new MeanLeafDepth(),
                new LeafDepthVariance(),
                new B1Index(),
                new B2Index(),
                new TotalCopheneticIndex(),
                new MeanMrcaDepth(),

                // tempo of diversification
                new PybusHarveyGamma(),
                new LineageAccumulationSlope(),

                // pairwise patristic geometry
                new TreeDiameter(),
                new PairwisePatristicDistanceVariance(),
                new MinPairwisePatristicDistance(),

                // evolutionary distinctiveness
                new EvolutionaryDistinctivenessCV());
    }

    @Override
    public int getNumImmutable() {
        return this.descriptors.size();
    }

    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        double[] description = new double[this.getNumImmutable()];

        for (int i = 0; i < this.getNumImmutable(); i++) {
            description[i] = this.descriptors.get(i).getDescription(this.tree);
        }

        return description;
     }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double update(double[] mutable, int nodeId) {
        return 0.0;
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0.0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    // ------------------------------------------------------------------
    // descriptor contract
    // ------------------------------------------------------------------

    private interface TreeDescriptor {
        double getDescription(Tree tree);
    }

    // ==================================================================
    // size
    // ==================================================================

    private static class TreeHeight implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return tree.getRoot().getHeight();
        }
    }

    // ==================================================================
    // branch-length structure
    // ==================================================================

    private static class TreeLength implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return treeLength(tree);
        }
    }

    private static class MeanBranchLength implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return mean(branchLengths(tree));
        }
    }

    private static class MaxBranchLength implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return max(branchLengths(tree));
        }
    }

    private static class MinBranchLength implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return min(branchLengths(tree));
        }
    }

    private static class BranchLengthVariance implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return variance(branchLengths(tree));
        }
    }

    /** Standard deviation of branch lengths divided by their mean; a scale-free dispersion measure. */
    private static class BranchLengthCoefficientOfVariation implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double[] lengths = branchLengths(tree);
            double m = mean(lengths);
            return Math.sqrt(variance(lengths)) / m;
        }
    }

    private static class BranchLengthSkewness implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return skewness(branchLengths(tree));
        }
    }

    private static class BranchLengthKurtosis implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return excessKurtosis(branchLengths(tree));
        }
    }

    /** Gini coefficient of branch lengths; zero when all branches are equal, approaching one when a few branches dominate. */
    private static class BranchLengthGini implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return gini(branchLengths(tree));
        }
    }

    /** Shannon entropy of the branch lengths treated as a probability distribution over total tree length. */
    private static class BranchLengthEntropy implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return shannonEntropy(branchLengths(tree));
        }
    }

    private static class MeanExternalBranchLength implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return mean(externalBranchLengths(tree));
        }
    }

    private static class MeanInternalBranchLength implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return mean(internalBranchLengths(tree));
        }
    }

    /** Ratio of the total internal branch length to the total external branch length. */
    private static class InternalToExternalBranchLengthRatio implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double external = sum(externalBranchLengths(tree));
            double internal = sum(internalBranchLengths(tree));
            return internal / external;
        }
    }

    /** Fraction of the total tree length that sits on internal branches (Lanyon's treeness). */
    private static class Treeness implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double total = treeLength(tree);
            double internal = sum(internalBranchLengths(tree));
            return internal / total;
        }
    }

    /**
     * Mean Fiala-Sokal stemminess over internal non-root nodes.
     * For a node the stemminess is the length of its parent branch divided by the sum of that
     * branch and every branch in the subtree it roots. Values near one indicate long stems and
     * short crowns.
     */
    private static class MeanStemminess implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            List<Double> values = new ArrayList<>();

            for (Node node : tree.getInternalNodes()) {
                if (node.isRoot()) {
                    continue;
                }

                double stem = node.getLength();
                double denominator = stem + subtreeLength(node);
                if (denominator > 0.0) {
                    values.add(stem / denominator);
                }
            }

            return mean(toArray(values));
        }
    }

    // ==================================================================
    // node-height (time) structure
    // ==================================================================

    private static class MeanInternalNodeHeight implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return mean(internalNodeHeights(tree));
        }
    }

    private static class InternalNodeHeightVariance implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return variance(internalNodeHeights(tree));
        }
    }

    /**
     * Variance of the root-to-tip distances.
     * Zero for a strictly clock-like (ultrametric) tree, growing as tip sampling times or rate
     * variation pull the leaves away from a common depth.
     */
    private static class RootToTipDistanceVariance implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double rootHeight = tree.getRoot().getHeight();
            double[] distances = leafHeights(tree);

            for (int i = 0; i < distances.length; i++) {
                distances[i] = rootHeight - distances[i];
            }

            return variance(distances);
        }
    }

    /** Length of the interval between the root and the next-oldest internal node, relative to tree height. */
    private static class RelativeTimeToFirstSplit implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double[] heights = internalNodeHeights(tree);
            if (heights.length < 2) {
                return 0.0;
            }

            Arrays.sort(heights);
            double rootHeight = heights[heights.length - 1];
            double secondHeight = heights[heights.length - 2];

            return rootHeight > 0.0 ? (rootHeight - secondHeight) / rootHeight : 0.0;
        }
    }

    /** Ratio of the shorter to the longer of the two branches descending from the root. */
    private static class BasalBranchLengthRatio implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            List<Node> children = tree.getRoot().getChildren();
            if (children.size() < 2) {
                return 0.0;
            }

            double shorter = Double.POSITIVE_INFINITY;
            double longer = 0.0;
            for (Node child : children) {
                double length = child.getLength();
                shorter = Math.min(shorter, length);
                longer = Math.max(longer, length);
            }

            return longer > 0.0 ? shorter / longer : 0.0;
        }
    }

    // ==================================================================
    // topological balance
    // ==================================================================

    /**
     * Colless imbalance index: summed absolute difference in descendant-leaf counts between the two
     * children of every bifurcating internal node. Polytomies contribute the spread between their
     * most and least populous child.
     */
    private static class CollessIndex implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return collessSum(tree);
        }
    }

    /** Colless index rescaled to [0, 1] by the maximum attainable for the leaf count (a caterpillar tree). */
    private static class NormalizedCollessIndex implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            int n = tree.getLeafNodeCount();
            if (n < 3) {
                return 0.0;
            }

            double maximum = (n - 1) * (n - 2) / 2.0;

            return collessSum(tree) / maximum;
        }
    }

    /** Sackin index: the sum over leaves of the number of edges on the path back to the root. */
    private static class SackinIndex implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return sum(intToDouble(leafDepths(tree)));
        }
    }

    private static class MeanLeafDepth implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return mean(intToDouble(leafDepths(tree)));
        }
    }

    private static class LeafDepthVariance implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return variance(intToDouble(leafDepths(tree)));
        }
    }

    /** Shao and Sokal's B1: sum over internal non-root nodes of the reciprocal of the longest downward path to a leaf. */
    private static class B1Index implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double total = 0.0;

            for (Node node : tree.getInternalNodes()) {
                if (node.isRoot()) {
                    continue;
                }

                int height = maxDepthToLeaf(node);
                if (height > 0) {
                    total += 1.0 / height;
                }
            }

            return total;
        }
    }

    /** Shao and Sokal's B2: sum over leaves of depth divided by two raised to that depth. */
    private static class B2Index implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double total = 0.0;

            for (int depth : leafDepths(tree)) {
                total += depth / Math.pow(2.0, depth);
            }

            return total;
        }
    }

    /**
     * Total cophenetic index: the sum over all leaf pairs of the depth of their most recent common
     * ancestor, computed by the equivalent per-node decomposition.
     */
    private static class TotalCopheneticIndex implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return totalCopheneticIndex(tree);
        }
    }

    /** Total cophenetic index divided by the number of leaf pairs; the mean common-ancestor depth. */
    private static class MeanMrcaDepth implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            int n = tree.getLeafNodeCount();
            if (n < 2) {
                return 0.0;
            }

            double pairs = n * (n - 1) / 2.0;

            return totalCopheneticIndex(tree) / pairs;
        }
    }

    // ==================================================================
    // tempo of diversification
    // ==================================================================

    /**
     * Pybus and Harvey's gamma statistic on the internal-node waiting times.
     *Negative values indicate that most branching happened early (a slowdown), positive values
     * that branching is concentrated near the present. Returns zero when there are too few leaves
     * for the statistic to be defined.
     */
    private static class PybusHarveyGamma implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            int n = tree.getLeafNodeCount();
            if (n < 4) {
                return 0.0;
            }

            // internal node heights sorted from the root downwards
            double[] heights = internalNodeHeights(tree);
            Arrays.sort(heights);
            reverse(heights);

            double present = min(leafHeights(tree));

            // g[j] is the time spent with j lineages, for j = 2 .. n
            double[] g = new double[n + 1];
            for (int j = 2; j <= n - 1; j++) {
                g[j] = heights[j - 2] - heights[j - 1];
            }
            g[n] = heights[n - 2] - present;

            double t = 0.0;
            for (int j = 2; j <= n; j++) {
                t += j * g[j];
            }
            if (t <= 0.0) {
                return 0.0;
            }

            double inner = 0.0;
            for (int i = 2; i <= n - 1; i++) {
                double partial = 0.0;
                for (int k = 2; k <= i; k++) {
                    partial += k * g[k];
                }
                inner += partial;
            }

            double numerator = inner / (n - 2) - t / 2.0;
            double denominator = t * Math.sqrt(1.0 / (12.0 * (n - 2)));

            return denominator > 0.0 ? numerator / denominator : 0.0;
        }
    }

    /**
     * Slope of a least-squares line through the lineage-through-time curve on a log scale.
     *
     * <p>Fitted to the points (branching time, natural log of the lineage count just after that
     * branching). Under a pure-birth process this estimates the net diversification rate.
     */
    private static class LineageAccumulationSlope implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double[] heights = internalNodeHeights(tree);
            if (heights.length < 2) {
                return 0.0;
            }

            Arrays.sort(heights);
            reverse(heights);
            double rootHeight = heights[0];

            double[] x = new double[heights.length];
            double[] y = new double[heights.length];
            for (int i = 0; i < heights.length; i++) {
                x[i] = rootHeight - heights[i];
                y[i] = Math.log(i + 2.0);
            }

            return slope(x, y);
        }
    }

    // ==================================================================
    // pairwise patristic geometry
    // ==================================================================

    private static class TreeDiameter implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double[] distances = pairwisePatristicDistances(tree);

            return distances.length == 0 ? 0.0 : max(distances);
        }
    }

    private static class PairwisePatristicDistanceVariance implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            return variance(pairwisePatristicDistances(tree));
        }
    }

    private static class MinPairwisePatristicDistance implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double[] distances = pairwisePatristicDistances(tree);

            return distances.length == 0 ? 0.0 : min(distances);
        }
    }

    // ==================================================================
    // evolutionary distinctiveness
    // ==================================================================

    /** Coefficient of variation of the fair-proportion evolutionary distinctiveness scores across leaves. */
    private static class EvolutionaryDistinctivenessCV implements TreeDescriptor {
        @Override
        public double getDescription(Tree tree) {
            double[] scores = evolutionaryDistinctiveness(tree);
            double m = mean(scores);

            return m > 0.0 ? Math.sqrt(variance(scores)) / m : 0.0;
        }
    }

    // ==================================================================
    // tree-derived arrays
    // ==================================================================

    /** Returns the length of every branch in the tree, that is every node except the root. */
    private static double[] branchLengths(Tree tree) {
        Node[] nodes = tree.getNodesAsArray();
        List<Double> lengths = new ArrayList<>(nodes.length);

        for (Node node : nodes) {
            if (!node.isRoot()) {
                lengths.add(node.getLength());
            }
        }

        return toArray(lengths);
    }

    /** Returns the length of every pendant branch, that is every branch leading to a leaf. */
    private static double[] externalBranchLengths(Tree tree) {
        List<Node> leaves = tree.getExternalNodes();
        double[] lengths = new double[leaves.size()];

        for (int i = 0; i < leaves.size(); i++) {
            lengths[i] = leaves.get(i).getLength();
        }

        return lengths;
    }

    /** Returns the length of every internal branch, that is every branch between two internal nodes. */
    private static double[] internalBranchLengths(Tree tree) {
        List<Node> internals = tree.getInternalNodes();
        List<Double> lengths = new ArrayList<>(internals.size());

        for (Node node : internals) {
            if (!node.isRoot()) {
                lengths.add(node.getLength());
            }
        }

        return toArray(lengths);
    }

    private static double[] internalNodeHeights(Tree tree) {
        List<Node> internals = tree.getInternalNodes();
        double[] heights = new double[internals.size()];

        for (int i = 0; i < internals.size(); i++) {
            heights[i] = internals.get(i).getHeight();
        }

        return heights;
    }

    private static double[] leafHeights(Tree tree) {
        List<Node> leaves = tree.getExternalNodes();
        double[] heights = new double[leaves.size()];

        for (int i = 0; i < leaves.size(); i++) {
            heights[i] = leaves.get(i).getHeight();
        }

        return heights;
    }

    /** Returns, for every leaf, the number of edges on the path from that leaf to the root. */
    private static int[] leafDepths(Tree tree) {
        List<Node> leaves = tree.getExternalNodes();
        int[] depths = new int[leaves.size()];

        for (int i = 0; i < leaves.size(); i++) {
            int depth = 0;
            for (Node node = leaves.get(i); !node.isRoot(); node = node.getParent()) {
                depth++;
            }
            depths[i] = depth;
        }

        return depths;
    }

    private static double treeLength(Tree tree) {
        return sum(branchLengths(tree));
    }

    /** Returns the summed branch length of the subtree rooted at the given node, excluding its parent branch. */
    private static double subtreeLength(Node node) {
        double total = 0.0;

        for (Node child : node.getChildren()) {
            total += child.getLength() + subtreeLength(child);
        }

        return total;
    }

    private static int descendantLeafCount(Node node) {
        if (node.isLeaf()) {
            return 1;
        }

        int count = 0;
        for (Node child : node.getChildren()) {
            count += descendantLeafCount(child);
        }

        return count;
    }

    private static int maxDepthToLeaf(Node node) {
        if (node.isLeaf()) {
            return 0;
        }

        int best = 0;
        for (Node child : node.getChildren()) {
            best = Math.max(best, maxDepthToLeaf(child));
        }

        return best + 1;
    }

    private static double collessSum(Tree tree) {
        double total = 0.0;

        for (Node node : tree.getInternalNodes()) {
            List<Node> children = node.getChildren();
            if (children.size() < 2) {
                continue;
            }

            int smallest = Integer.MAX_VALUE;
            int largest = 0;
            for (Node child : children) {
                int leaves = descendantLeafCount(child);
                smallest = Math.min(smallest, leaves);
                largest = Math.max(largest, leaves);
            }
            total += largest - smallest;
        }

        return total;
    }

    private static double totalCopheneticIndex(Tree tree) {
        double total = 0.0;

        for (Node node : tree.getInternalNodes()) {
            for (Node child : node.getChildren()) {
                int leaves = descendantLeafCount(child);
                total += leaves * (leaves - 1) / 2.0;
            }
        }

        return total;
    }

    /**
     * Returns the patristic (path) distance between every pair of leaves.
     *
     * <p>The distance for a pair is recovered from the height of their common ancestor as
     * {@code 2 * mrcaHeight - heightA - heightB}, which telescopes to the sum of the intervening
     * branch lengths for any tree whose branch lengths are height differences.
     */
    private static double[] pairwisePatristicDistances(Tree tree) {
        List<Node> leaves = tree.getExternalNodes();
        int n = leaves.size();
        if (n < 2) {
            return new double[0];
        }

        double[] distances = new double[n * (n - 1) / 2];
        int index = 0;
        for (int i = 0; i < n; i++) {
            Node a = leaves.get(i);
            Set<Node> ancestors = new HashSet<>();
            for (Node node = a; node != null; node = node.getParent()) {
                ancestors.add(node);
            }

            for (int j = i + 1; j < n; j++) {
                Node b = leaves.get(j);
                Node mrca = b;
                while (mrca != null && !ancestors.contains(mrca)) {
                    mrca = mrca.getParent();
                }

                double mrcaHeight = mrca != null ? mrca.getHeight() : tree.getRoot().getHeight();
                distances[index++] = 2.0 * mrcaHeight - a.getHeight() - b.getHeight();
            }
        }

        return distances;
    }

    /**
     * Returns the fair-proportion evolutionary distinctiveness of every leaf.
     *
     * <p>Each branch shares its length equally among the leaves it subtends; a leaf's score is the
     * sum of the shares it collects on the path to the root. The scores sum to the total tree
     * length.
     */
    private static double[] evolutionaryDistinctiveness(Tree tree) {
        List<Node> leaves = tree.getExternalNodes();
        double[] scores = new double[leaves.size()];

        for (int i = 0; i < leaves.size(); i++) {
            double score = 0.0;
            for (Node node = leaves.get(i); !node.isRoot(); node = node.getParent()) {
                score += node.getLength() / descendantLeafCount(node);
            }
            scores[i] = score;
        }

        return scores;
    }

    // ==================================================================
    // numeric helpers
    // ==================================================================

    private static double sum(double[] values) {
        double total = 0.0;
        for (double value : values) {
            total += value;
        }

        return total;
    }

    private static double mean(double[] values) {
        return values.length == 0 ? 0.0 : sum(values) / values.length;
    }

    /** Population variance; returns zero for fewer than two values. */
    private static double variance(double[] values) {
        if (values.length < 2) {
            return 0.0;
        }

        double m = mean(values);
        double squaredError = 0.0;
        for (double value : values) {
            double delta = value - m;
            squaredError += delta * delta;
        }

        return squaredError / values.length;
    }

    /** Fisher-Pearson skewness; returns zero when the values have no spread. */
    private static double skewness(double[] values) {
        if (values.length < 3) {
            return 0.0;
        }

        double m = mean(values);
        double sd = Math.sqrt(variance(values));
        if (sd == 0.0) {
            return 0.0;
        }

        double thirdMoment = 0.0;
        for (double value : values) {
            double z = (value - m) / sd;
            thirdMoment += z * z * z;
        }

        return thirdMoment / values.length;
    }

    /** Excess kurtosis (normal distribution has zero); returns zero when the values have no spread. */
    private static double excessKurtosis(double[] values) {
        if (values.length < 4) {
            return 0.0;
        }

        double m = mean(values);
        double sd = Math.sqrt(variance(values));
        if (sd == 0.0) {
            return 0.0;
        }

        double fourthMoment = 0.0;
        for (double value : values) {
            double z = (value - m) / sd;
            fourthMoment += z * z * z * z;
        }

        return fourthMoment / values.length - 3.0;
    }

    private static double median(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }

        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int mid = sorted.length / 2;

        return sorted.length % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2.0 : sorted[mid];
    }

    private static double min(double[] values) {
        double smallest = Double.POSITIVE_INFINITY;
        for (double value : values) {
            smallest = Math.min(smallest, value);
        }

        return smallest == Double.POSITIVE_INFINITY ? 0.0 : smallest;
    }

    private static double max(double[] values) {
        double largest = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            largest = Math.max(largest, value);
        }

        return largest == Double.NEGATIVE_INFINITY ? 0.0 : largest;
    }

    /**
     * Gini coefficient of a set of non-negative values.
     *
     * <p>Zero when every value is equal, approaching one as the total concentrates onto a single
     * value. Returns zero when the values sum to zero.
     */
    private static double gini(double[] values) {
        int n = values.length;
        if (n < 2) {
            return 0.0;
        }

        double total = sum(values);
        if (total <= 0.0) {
            return 0.0;
        }

        double absoluteDifferences = 0.0;
        for (double a : values) {
            for (double b : values) {
                absoluteDifferences += Math.abs(a - b);
            }
        }

        return absoluteDifferences / (2.0 * n * n * (total / n));
    }

    /** Shannon entropy in nats of a set of non-negative values normalised to a probability distribution. */
    private static double shannonEntropy(double[] values) {
        double total = sum(values);
        if (total <= 0.0) {
            return 0.0;
        }

        double entropy = 0.0;
        for (double value : values) {
            if (value > 0.0) {
                double p = value / total;
                entropy -= p * Math.log(p);
            }
        }

        return entropy;
    }

    /** Least-squares slope of y on x; returns zero when x has no spread. */
    private static double slope(double[] x, double[] y) {
        int n = x.length;
        if (n < 2) {
            return 0.0;
        }

        double meanX = mean(x);
        double meanY = mean(y);
        double covariance = 0.0;
        double varianceX = 0.0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - meanX;
            covariance += dx * (y[i] - meanY);
            varianceX += dx * dx;
        }

        return varianceX > 0.0 ? covariance / varianceX : 0.0;
    }

    private static void reverse(double[] values) {
        for (int i = 0, j = values.length - 1; i < j; i++, j--) {
            double temp = values[i];
            values[i] = values[j];
            values[j] = temp;
        }
    }

    private static double[] intToDouble(int[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }

        return result;
    }

    private static double[] toArray(List<Double> values) {
        double[] result = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            result[i] = values.get(i);
        }

        return result;
    }

}
