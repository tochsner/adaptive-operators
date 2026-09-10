package surrogate;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * Tree metric counting the clades that two trees do not share.
 *
 * <p>A tree is embedded as the sorted hashes of its clades, one per internal node, and the distance
 * between two trees is the size of the symmetric difference of those two sets divided by the
 * largest value it could take. Distances therefore lie in [0, 1], with 0 for trees of identical
 * shape and 1 for trees sharing no clade beyond the root.
 *
 * <p>Clades are rooted, so this compares the trees as rooted rather than reducing them to
 * bipartitions. Branch lengths are ignored entirely, which makes the metric invariant to any
 * rescaling of a tree for free, but also blind to how the trees differ in time: two trees with the
 * same shape and wildly different node heights are at distance zero.
 *
 * <p>Taxa are identified by leaf node number, so all compared trees must share a taxon numbering.
 *
 * <p>A clade is identified by the exclusive or of a random value per taxon in it, so two distinct
 * clades collide with probability 2^-64. Unlike {@link PairwiseDistanceTreeMetric} the embedding is
 * exact rather than a subsample, so there is no accuracy knob to set.
 */
public class RobinsonFouldsTreeMetric {

    private final long[] taxonHash;

    public RobinsonFouldsTreeMetric(int numTaxa) {
        this.taxonHash = new long[numTaxa];

        for (int i = 0; i < numTaxa; i++) {
            this.taxonHash[i] = Randomizer.nextLong();
        }
    }

    /**
     * Embed a tree as the sorted hashes of its clades, one per internal node.
     */
    public long[] embed(Tree tree) {
        long[] hashOfNode = new long[tree.getNodeCount()];
        long[] cladeHashes = new long[tree.getInternalNodeCount()];

        int index = 0;
        for (Node node : postOrder(tree)) {
            if (node.isLeaf()) {
                hashOfNode[node.getNr()] = this.taxonHash[node.getNr()];
                continue;
            }

            // children are visited first, so the clade below a node is already summarised by the
            // time the node itself comes up. Exclusive or ignores the order the children arrive in.
            long cladeHash = 0L;
            for (Node child : node.getChildren()) {
                cladeHash ^= hashOfNode[child.getNr()];
            }

            hashOfNode[node.getNr()] = cladeHash;
            cladeHashes[index++] = cladeHash;
        }

        Arrays.sort(cladeHashes);

        return cladeHashes;
    }

    /**
     * Robinson-Foulds distance between two trees, scaled into [0, 1].
     */
    public double distance(Tree firstTree, Tree secondTree) {
        return distance(this.embed(firstTree), this.embed(secondTree));
    }

    /**
     * Robinson-Foulds distance between two embeddings produced by the same metric, scaled into
     * [0, 1] by the largest number of clades that could have differed.
     */
    public static double distance(long[] firstEmbedding, long[] secondEmbedding) {
        // both trees contain the clade of every taxon at their root, so that one can never differ
        int differingClades = countDifferingClades(firstEmbedding, secondEmbedding);
        int mostThatCouldDiffer = firstEmbedding.length + secondEmbedding.length - 2;

        if (mostThatCouldDiffer <= 0) {
            return 0.0;
        }

        return (double) differingClades / mostThatCouldDiffer;
    }

    /**
     * Robinson-Foulds distance between two embeddings as a raw count of clades, without scaling.
     */
    public static int countDifferingClades(long[] firstEmbedding, long[] secondEmbedding) {
        int shared = 0;
        int first = 0;
        int second = 0;

        // a merge over the two sorted hash lists, counting the ones they have in common
        while (first < firstEmbedding.length && second < secondEmbedding.length) {
            if (firstEmbedding[first] == secondEmbedding[second]) {
                shared++;
                first++;
                second++;
            } else if (firstEmbedding[first] < secondEmbedding[second]) {
                first++;
            } else {
                second++;
            }
        }

        return firstEmbedding.length + secondEmbedding.length - 2 * shared;
    }

    /**
     * Lists the nodes of a tree so that every node comes after all of its descendants.
     */
    private static Node[] postOrder(Tree tree) {
        Node[] order = new Node[tree.getNodeCount()];
        Deque<Node> pending = new ArrayDeque<>();
        pending.push(tree.getRoot());

        // filling the array back to front turns a walk that reaches parents first into one that
        // hands them back last, without recursing over a tree that may be deep
        int index = order.length;
        while (!pending.isEmpty()) {
            Node node = pending.pop();
            order[--index] = node;

            for (Node child : node.getChildren()) {
                pending.push(child);
            }
        }

        return order;
    }
}
