package surrogate;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Tree metric based on a fixed random subset of taxon pairs.
 *
 * <p>Embeddings are scaled to unit length, which makes the metric invariant to a global rescaling
 * of a tree and so lets trees measured on different scales -- substitutions per site versus real
 * time, say -- be compared directly. Only the relative shape of a tree is retained; two trees that
 * differ solely by a constant factor are at distance zero. Distances therefore lie in [0, 2].
 */
public class PairwiseDistanceTreeMetric {

    private final int[] firstTaxon;
    private final int[] secondTaxon;

    // marks the root path of a pair's first taxon; stamped per pair so it never needs clearing
    private int[] visitStamp = new int[0];
    private int currentStamp;

    public PairwiseDistanceTreeMetric(int numTaxa, int numPairs) {
        int maxNumPairs = numTaxa * (numTaxa - 1) / 2;
        int numSampledPairs = Math.min(numPairs, maxNumPairs);

        this.firstTaxon = new int[numSampledPairs];
        this.secondTaxon = new int[numSampledPairs];

        this.samplePairs(numTaxa, numSampledPairs);
    }

    private void samplePairs(int numTaxa, int numSampledPairs) {
        Set<Long> sampledPairs = new HashSet<>();

        while (sampledPairs.size() < numSampledPairs) {
            int first = Randomizer.nextInt(numTaxa);

            // draw from the remaining taxa and shift past the first one, so the two never collide
            int second = Randomizer.nextInt(numTaxa - 1);
            if (second >= first) {
                second++;
            }

            int lowTaxon = Math.min(first, second);
            int highTaxon = Math.max(first, second);

            if (sampledPairs.add((long) lowTaxon * numTaxa + highTaxon)) {
                int pairIndex = sampledPairs.size() - 1;
                this.firstTaxon[pairIndex] = lowTaxon;
                this.secondTaxon[pairIndex] = highTaxon;
            }
        }
    }

    /**
     * Embed a tree as the vector of patristic distances over the fixed taxon pairs, scaled to unit
     * length so that the embedding does not depend on the overall scale of the tree.
     */
    public double[] embed(Tree tree) {
        if (this.visitStamp.length < tree.getNodeCount()) {
            this.visitStamp = new int[tree.getNodeCount()];
        }

        double[] embedding = new double[this.firstTaxon.length];

        for (int i = 0; i < this.firstTaxon.length; i++) {
            Node first = tree.getNode(this.firstTaxon[i]);
            Node second = tree.getNode(this.secondTaxon[i]);
            Node commonAncestor = this.getCommonAncestor(first, second);

            // every branch length is a height difference, so summing them along the path
            // collapses to the two legs hanging below the common ancestor
            embedding[i] = 2.0 * commonAncestor.getHeight() - first.getHeight() - second.getHeight();
        }

        return normalize(embedding);
    }

    /**
     * Scale an embedding to unit length, in place. A global rescaling of a tree multiplies every
     * patristic distance by the same factor, so dividing out the norm cancels that factor exactly.
     */
    private static double[] normalize(double[] embedding) {
        double sumOfSquares = 0.0;
        for (double pairDistance : embedding) {
            sumOfSquares += pairDistance * pairDistance;
        }

        // a tree whose sampled pairs are all at distance zero has no shape to preserve
        double norm = Math.sqrt(sumOfSquares);
        if (norm == 0.0) {
            return embedding;
        }

        for (int i = 0; i < embedding.length; i++) {
            embedding[i] /= norm;
        }

        return embedding;
    }

    /**
     * Euclidean distance between the embeddings of two trees.
     */
    public double distance(Tree firstTree, Tree secondTree) {
        return distance(this.embed(firstTree), this.embed(secondTree));
    }

    /**
     * Euclidean distance between two embeddings produced by the same metric.
     */
    public static double distance(double[] firstEmbedding, double[] secondEmbedding) {
        if (firstEmbedding.length != secondEmbedding.length) {
            throw new IllegalArgumentException(
                    "Embeddings have different lengths: " + firstEmbedding.length
                            + " and " + secondEmbedding.length + ".");
        }

        double sumOfSquares = 0.0;
        for (int i = 0; i < firstEmbedding.length; i++) {
            double difference = firstEmbedding[i] - secondEmbedding[i];
            sumOfSquares += difference * difference;
        }

        return Math.sqrt(sumOfSquares);
    }

    private Node getCommonAncestor(Node first, Node second) {
        this.currentStamp++;

        // stamps must stay strictly ahead of everything already in the buffer, so wrap by clearing
        if (this.currentStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.visitStamp, 0);
            this.currentStamp = 1;
        }

        for (Node current = first; current != null; current = current.getParent()) {
            this.visitStamp[current.getNr()] = this.currentStamp;
        }

        Node commonAncestor = second;
        while (commonAncestor != null && this.visitStamp[commonAncestor.getNr()] != this.currentStamp) {
            commonAncestor = commonAncestor.getParent();
        }

        if (commonAncestor == null) {
            throw new IllegalArgumentException("Nodes are not part of the same tree.");
        }

        return commonAncestor;
    }
}
