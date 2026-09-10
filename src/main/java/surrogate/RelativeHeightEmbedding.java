package surrogate;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Encodes a tree as its internal node heights, sorted ascending and divided by the root height.
 * Embeddings live in the order simplex {@code 0 <= s_1 <= ... <= s_{n-1} = 1}, extended by a
 * lower bound per entry when tips are dated rather than contemporaneous.
 */
public final class RelativeHeightEmbedding {

    /** Ordering gaps this small count as ties, whose order {@link #decode} is free to settle. */
    private static final double TIE_TOLERANCE = 1e-12;

    private RelativeHeightEmbedding() {
    }

    /**
     * Encodes a tree as its internal node heights, sorted ascending and divided by the root height.
     */
    public static double[] encode(Tree tree) {
        double rootHeight = tree.getRoot().getHeight();
        double[] relativeHeights = new double[tree.getInternalNodeCount()];

        int index = 0;
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                relativeHeights[index++] = node.getHeight() / rootHeight;
            }
        }

        Arrays.sort(relativeHeights);

        return relativeHeights;
    }

    /**
     * Installs the given topology on the target tree and writes the embedding back onto it, scaled
     * by the given root height.
     *
     * <p>The heights are assigned in rank order, so the k-th smallest value lands on the internal
     * node that ranks k-th by height in the topology. Because the topology already has every parent
     * at or above its children, a sorted embedding reproduces that ordering and the result is a
     * valid tree without any further checking.
     */
    public static void decode(Tree target, Tree topology, double[] relativeHeights, double rootHeight) {
        // assignFromFragile reuses the target's existing Node objects, unlike assignFrom, which
        // reallocates the node arrays and would invalidate the state's store/restore backups
        target.assignFromFragile(topology);

        // a step is allowed to cross tied entries, which segmentBounds deliberately does not
        // prevent, so settle their order here before assigning by rank
        double[] sortedHeights = relativeHeights.clone();
        Arrays.sort(sortedHeights);

        Node[] internalNodes = internalNodesByHeight(target);

        for (int rank = 0; rank < internalNodes.length; rank++) {
            internalNodes[rank].setHeight(sortedHeights[rank] * rootHeight);
        }
    }

    /**
     * Returns, per internal node rank of the given topology, the smallest relative height that node
     * may take without dropping below a tip hanging directly off it.
     * This is all zeroes when the tips are contemporaneous, and only bites for dated tips.
     */
    public static double[] minimumRelativeHeights(Tree topology, double rootHeight) {
        Node[] internalNodes = internalNodesByHeight(topology);
        double[] minima = new double[internalNodes.length];

        for (int rank = 0; rank < internalNodes.length; rank++) {
            double highestTip = 0.0;

            for (Node child : internalNodes[rank].getChildren()) {
                if (child.isLeaf()) {
                    highestTip = Math.max(highestTip, child.getHeight());
                }
            }

            minima[rank] = highestTip / rootHeight;
        }

        return minima;
    }

    /**
     * Returns the interval of step sizes for which {@code anchor + step * direction} stays a valid
     * embedding, as a two element array holding its lower and upper end.
     *
     * <p>Both the ordering constraints and the per entry lower bounds are affine in the step size,
     * so each one cuts off a half line and the admissible set is their intersection. A zero step is
     * always admissible because the anchor is itself a valid embedding, so the interval is never
     * empty, and the order simplex is bounded so it is never infinite either.
     */
    public static double[] segmentBounds(double[] anchor, double[] direction, double[] minima) {
        double lower = Double.NEGATIVE_INFINITY;
        double upper = Double.POSITIVE_INFINITY;

        // every internal node has to stay above the tips hanging directly off it
        for (int i = 0; i < anchor.length; i++) {
            double slack = anchor[i] - minima[i];

            if (direction[i] > 0.0) {
                lower = Math.max(lower, -slack / direction[i]);
            } else if (direction[i] < 0.0) {
                upper = Math.min(upper, -slack / direction[i]);
            }
        }

        // consecutive heights have to keep their order
        for (int i = 0; i + 1 < anchor.length; i++) {
            double gap = anchor[i + 1] - anchor[i];
            double slope = direction[i + 1] - direction[i];

            // tied entries are interchangeable, and holding their order would pin the step to zero
            // from one side. Two ties of opposite slope would then pin it from both and leave no
            // interval at all, so let them cross and have decode settle the order by sorting.
            if (gap <= TIE_TOLERANCE) {
                continue;
            }

            if (slope > 0.0) {
                lower = Math.max(lower, -gap / slope);
            } else if (slope < 0.0) {
                upper = Math.min(upper, -gap / slope);
            }
        }

        return new double[] { lower, upper };
    }

    private static Node[] internalNodesByHeight(Tree tree) {
        Node[] internalNodes = new Node[tree.getInternalNodeCount()];

        int index = 0;
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                internalNodes[index++] = node;
            }
        }

        int[] depthOfNode = new int[tree.getNodeCount()];
        for (Node node : internalNodes) {
            int depth = 0;
            for (Node ancestor = node; ancestor.getParent() != null; ancestor = ancestor.getParent()) {
                depth++;
            }
            depthOfNode[node.getNr()] = depth;
        }

        // ties are broken deepest first, so a descendant always ranks before its ancestor. Tied
        // nodes can be a parent and child joined by a zero length branch, and assigning ascending
        // values in the other order would push the child above its parent.
        Comparator<Node> byHeightThenDepth = Comparator
                .comparingDouble(Node::getHeight)
                .thenComparingInt(node -> -depthOfNode[node.getNr()]);
        Arrays.sort(internalNodes, byHeightThenDepth);

        return internalNodes;
    }
}
