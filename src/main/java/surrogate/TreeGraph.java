package surrogate;

import beast.base.evolution.tree.Tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gives access to the MST of a set of sampled trees as described by the paper
 * Graph-accelerated Markov Chain Monte Carlo using Approximate Samples (Duan &
 * Bhattacharya, 2025, https://www.jmlr.org/papers/volume26/24-1024/24-1024.pdf).
 */
public class TreeGraph {

    RobinsonFouldsTreeMetric treeMetric;
    double kappa = 1.0;

    private List<SampledTree> samples;
    private long[][] embeddings;
    private double[][] heightEmbeddings;
    private List<List<Integer>> adjacency;
    private Map<SampledTree, Integer> indexOfSample;

    public TreeGraph(List<SampledTree> topologySamples) {
        // define metric

        int numTaxa = topologySamples.getFirst().topology().getLeafNodeCount();
        this.treeMetric = new RobinsonFouldsTreeMetric(numTaxa);

        // build up graph

        this.buildGraph(topologySamples);
    }

    /**
     * Builds up the graph by computing the MST of the fully connected graph with the topologySamples
     * as vertices and the edge weights computed using {@code getEdgeWeight}.
     */
    private void buildGraph(List<SampledTree> topologySamples) {
        this.samples = List.copyOf(topologySamples);
        int numSamples = this.samples.size();

        this.indexOfSample = new HashMap<>();
        this.adjacency = new ArrayList<>(numSamples);

        for (int i = 0; i < numSamples; i++) {
            this.indexOfSample.put(this.samples.get(i), i);
            this.adjacency.add(new ArrayList<>());
        }

        // embed every sample once up front: the spanning tree inspects all O(n^2) edges, so
        // embedding inside the weight would redo the same work n times over
        this.embeddings = new long[numSamples][];
        this.heightEmbeddings = new double[numSamples][];
        for (int i = 0; i < numSamples; i++) {
            Tree topology = this.samples.get(i).topology();
            this.embeddings[i] = this.treeMetric.embed(topology);
            this.heightEmbeddings[i] = RelativeHeightEmbedding.encode(topology);
        }

        if (numSamples < 2) {
            return;
        }

        // Prim's algorithm, kept dense rather than heap-based because the underlying graph is
        // fully connected, which makes the O(n^2) form both simpler and faster
        boolean[] isInTree = new boolean[numSamples];
        double[] cheapestWeight = new double[numSamples];
        int[] cheapestEndpoint = new int[numSamples];

        Arrays.fill(cheapestWeight, Double.POSITIVE_INFINITY);
        Arrays.fill(cheapestEndpoint, -1);
        cheapestWeight[0] = 0.0;

        for (int step = 0; step < numSamples; step++) {
            // grow the tree by the cheapest edge leaving it
            int next = -1;
            for (int i = 0; i < numSamples; i++) {
                if (!isInTree[i] && (next == -1 || cheapestWeight[i] < cheapestWeight[next])) {
                    next = i;
                }
            }

            isInTree[next] = true;

            if (cheapestEndpoint[next] != -1) {
                this.adjacency.get(next).add(cheapestEndpoint[next]);
                this.adjacency.get(cheapestEndpoint[next]).add(next);
            }

            for (int i = 0; i < numSamples; i++) {
                if (isInTree[i]) {
                    continue;
                }

                double weight = this.getEdgeWeight(next, i);
                if (weight < cheapestWeight[i]) {
                    cheapestWeight[i] = weight;
                    cheapestEndpoint[i] = next;
                }
            }
        }
    }

    /**
     * Returns the sampled tree closest to tree according to the tree metric.
     *
     * <p>The Robinson-Foulds distance only sees topology, so several samples routinely tie for
     * closest, and which one comes back is decided by their order. Callers asking whether a
     * particular sample is a closest one want {@link #isClosest} rather than a comparison against
     * what this returns.
     */
    public SampledTree getClosest(Tree tree) {
        long[] embedding = this.treeMetric.embed(tree);

        SampledTree closest = null;
        double closestDistance = Double.POSITIVE_INFINITY;

        for (int i = 0; i < this.samples.size(); i++) {
            double distance = RobinsonFouldsTreeMetric.distance(embedding, this.embeddings[i]);

            if (distance < closestDistance) {
                closestDistance = distance;
                closest = this.samples.get(i);
            }
        }

        return closest;
    }

    /**
     * Returns whether no sampled tree is strictly closer to the given tree than the given candidate.
     */
    public boolean isClosest(Tree tree, SampledTree candidate) {
        long[] embedding = this.treeMetric.embed(tree);
        double candidateDistance = RobinsonFouldsTreeMetric.distance(
                embedding, this.embeddings[this.getIndex(candidate)]);

        for (long[] sampleEmbedding : this.embeddings) {
            if (RobinsonFouldsTreeMetric.distance(embedding, sampleEmbedding) < candidateDistance) {
                return false;
            }
        }

        return true;
    }

    /**
     * Returns a list with the neighbors of the sampled tree in the graph.
     */
    public List<Tree> getNeighbors(SampledTree tree) {
        List<Tree> neighbors = new ArrayList<>();

        for (int neighbor : this.adjacency.get(this.getIndex(tree))) {
            neighbors.add(this.samples.get(neighbor).topology());
        }

        return neighbors;
    }

    /**
     * Returns the degree of the sampled tree in the graph, which the Hastings ratio of a random
     * walk over the graph needs in order to correct for differing neighborhood sizes.
     */
    public int getDegree(SampledTree tree) {
        return this.adjacency.get(this.getIndex(tree)).size();
    }

    /**
     * Returns every sampled tree within the given number of hops of the given one, the centre
     * included.
     *
     * <p>Graph distance is symmetric, so one tree lies in another's ball exactly when the reverse
     * holds. The Hastings ratio of a jump between two balls relies on that.
     */
    public List<SampledTree> getBall(SampledTree tree, int radius) {
        boolean[] isInBall = new boolean[this.samples.size()];
        List<SampledTree> ball = new ArrayList<>();

        int centre = this.getIndex(tree);
        isInBall[centre] = true;
        ball.add(this.samples.get(centre));

        List<Integer> frontier = new ArrayList<>();
        frontier.add(centre);

        for (int hop = 0; hop < radius; hop++) {
            List<Integer> nextFrontier = new ArrayList<>();

            for (int node : frontier) {
                for (int neighbor : this.adjacency.get(node)) {
                    if (isInBall[neighbor]) {
                        continue;
                    }

                    isInBall[neighbor] = true;
                    nextFrontier.add(neighbor);
                    ball.add(this.samples.get(neighbor));
                }
            }

            frontier = nextFrontier;
        }

        return ball;
    }

    /**
     * Returns the cached relative height embedding of a sampled tree.
     */
    public double[] getHeightEmbedding(SampledTree tree) {
        return this.heightEmbeddings[this.getIndex(tree)];
    }

    private int getIndex(SampledTree tree) {
        Integer index = this.indexOfSample.get(tree);

        if (index == null) {
            throw new IllegalArgumentException("The given tree is not a node of this graph.");
        }

        return index;
    }

    private double getEdgeWeight(int firstSample, int secondSample) {
        double likelihoodDifference = Math.abs(
                this.samples.get(firstSample).logLikelihood()
                        - this.samples.get(secondSample).logLikelihood());

        if (likelihoodDifference < this.kappa) {
            double treeDistance = RobinsonFouldsTreeMetric.distance(
                    this.embeddings[firstSample], this.embeddings[secondSample]);
            return this.kappa / (1 + treeDistance);
        } else {
            return likelihoodDifference;
        }
    }

}
