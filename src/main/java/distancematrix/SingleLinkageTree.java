package distancematrix;

import java.util.Arrays;

/**
 * Maintains a symmetric distance matrix over taxa together with the single-linkage tree it induces.
 *
 * <p>Single-linkage clustering walks the taxon pairs in ascending order of distance and merges two
 * clusters whenever the endpoints of the current pair are not yet connected. That is exactly
 * Kruskal's algorithm, so the merges actually performed are the {@code n - 1} edges of a minimum
 * spanning tree of the complete graph on taxa, and every other matrix entry is skipped. The
 * resulting ultrametric is the maximal subdominant ultrametric of the matrix, which means the tree
 * is a valid rooted ultrametric time tree for any input matrix, and that an already ultrametric
 * matrix is reproduced exactly.
 *
 * <p>Because the tree only depends on the spanning tree, a change to a single matrix entry can be
 * applied incrementally instead of re-running the clustering from scratch.
 *
 * <p>Clusters are numbered like BEAST nodes: {@code 0 .. n-1} are the leaves and {@code n + k} is
 * the cluster created by the {@code k}-th merge, so {@code 2n - 2} is always the root.
 */
public class SingleLinkageTree {

    /** What a single-entry change did to the induced tree. */
    public enum UpdateKind {
        /** The entry sits outside the spanning tree and stayed there. The tree is untouched. */
        UNCHANGED,
        /** The spanning tree is the same and so is the merge order, so one node height moved. */
        HEIGHT_ONLY,
        /** The spanning tree or the merge order changed, so the whole dendrogram was rebuilt. */
        STRUCTURE
    }

    private final int leafCount;
    private final int pairCount;

    private final double[] distances;

    // minimum spanning tree, held as n - 1 edges plus a pair index -> edge slot lookup
    private final int[] edgeFirst;
    private final int[] edgeSecond;
    private final double[] edgeWeight;
    private final int[] edgeOfPair;

    // dendrogram, holding the n - 1 merges in ascending height order
    private final int[] mergeLeft;
    private final int[] mergeRight;
    private final double[] mergeHeight;
    private final int[] clusterSize;

    // spanning tree edges in ascending weight order, and the inverse lookup. The k-th entry of
    // edgeOrder is the edge that made the k-th merge, so rankOfEdge maps an edge to its merge.
    private final int[] edgeOrder;
    private final int[] rankOfEdge;

    private int lastChangedMerge = -1;

    // snapshot of everything above, so a rejected proposal can be rolled back
    private final int[] storedEdgeFirst;
    private final int[] storedEdgeSecond;
    private final double[] storedEdgeWeight;
    private final int[] storedEdgeOfPair;
    private final int[] storedMergeLeft;
    private final int[] storedMergeRight;
    private final double[] storedMergeHeight;
    private final int[] storedClusterSize;
    private final int[] storedEdgeOrder;
    private final int[] storedRankOfEdge;
    private final double[] storedDistances;
    private int storedPairIndex = -1;
    private double storedDistance;
    private boolean storedWholeMatrix = false;

    // scratch buffers reused across updates
    private final int[] adjacencyStart;
    private final int[] adjacencyNode;
    private final int[] adjacencyEdge;
    private final int[] queue;
    private final int[] maxEdgeOnPath;
    private final boolean[] visited;
    private final int[] parent;
    private final int[] clusterOf;

    public SingleLinkageTree(int leafCount, double[] distances) {
        if (leafCount < 3) {
            throw new IllegalArgumentException("SingleLinkageTree requires at least three taxa");
        }

        this.leafCount = leafCount;
        this.pairCount = leafCount * (leafCount - 1) / 2;

        if (distances.length != this.pairCount) {
            throw new IllegalArgumentException(
                    "expected " + this.pairCount + " distances but got " + distances.length);
        }

        this.distances = distances.clone();

        int edgeCount = leafCount - 1;
        this.edgeFirst = new int[edgeCount];
        this.edgeSecond = new int[edgeCount];
        this.edgeWeight = new double[edgeCount];
        this.edgeOfPair = new int[this.pairCount];

        this.mergeLeft = new int[edgeCount];
        this.mergeRight = new int[edgeCount];
        this.mergeHeight = new double[edgeCount];
        this.clusterSize = new int[2 * leafCount - 1];

        this.storedEdgeFirst = new int[edgeCount];
        this.storedEdgeSecond = new int[edgeCount];
        this.storedEdgeWeight = new double[edgeCount];
        this.storedEdgeOfPair = new int[this.pairCount];
        this.storedMergeLeft = new int[edgeCount];
        this.storedMergeRight = new int[edgeCount];
        this.storedMergeHeight = new double[edgeCount];
        this.storedClusterSize = new int[2 * leafCount - 1];
        this.storedEdgeOrder = new int[edgeCount];
        this.storedRankOfEdge = new int[edgeCount];
        this.storedDistances = new double[this.pairCount];

        this.adjacencyStart = new int[leafCount + 1];
        this.adjacencyNode = new int[2 * edgeCount];
        this.adjacencyEdge = new int[2 * edgeCount];
        this.queue = new int[leafCount];
        this.maxEdgeOnPath = new int[leafCount];
        this.visited = new boolean[leafCount];
        this.parent = new int[leafCount];
        this.clusterOf = new int[leafCount];
        this.edgeOrder = new int[edgeCount];
        this.rankOfEdge = new int[edgeCount];

        buildMinimumSpanningTree();
        rebuildDendrogram();
        snapshot();
    }

    /**
     * Replaces the whole matrix and rebuilds the spanning tree and the dendrogram from scratch.
     *
     * <p>Used when the matrix has to be rebuilt from the tree, which the incremental machinery
     * cannot express because more than one entry moves at once.
     */
    public void reset(double[] distances) {
        if (distances.length != this.pairCount) {
            throw new IllegalArgumentException(
                    "expected " + this.pairCount + " distances but got " + distances.length);
        }

        System.arraycopy(distances, 0, this.distances, 0, this.pairCount);
        buildMinimumSpanningTree();
        rebuildDendrogram();
        snapshot();
    }

    /** Twice the height of the most recent common ancestor of every taxon pair. */
    public void writeCopheneticMatrix(double[] cophenetic) {
        int[][] clusterLeaves = new int[2 * this.leafCount - 1][];
        for (int leaf = 0; leaf < this.leafCount; leaf++) {
            clusterLeaves[leaf] = new int[]{leaf};
        }

        for (int merge = 0; merge < this.leafCount - 1; merge++) {
            int[] left = clusterLeaves[this.mergeLeft[merge]];
            int[] right = clusterLeaves[this.mergeRight[merge]];
            double distance = 2.0 * this.mergeHeight[merge];

            for (int a : left) {
                for (int b : right) {
                    cophenetic[pairIndex(a, b)] = distance;
                }
            }

            int[] clade = new int[left.length + right.length];
            System.arraycopy(left, 0, clade, 0, left.length);
            System.arraycopy(right, 0, clade, left.length, right.length);
            clusterLeaves[this.leafCount + merge] = clade;
        }
    }

    public int getLeafCount() {
        return this.leafCount;
    }

    public int getPairCount() {
        return this.pairCount;
    }

    /**
     * Index of the pair {@code (i, j)} in the flattened upper triangle of the matrix.
     */
    public int pairIndex(int i, int j) {
        int first = Math.min(i, j);
        int second = Math.max(i, j);
        return first * this.leafCount - first * (first + 1) / 2 + (second - first - 1);
    }

    /** Whether the pair is one of the {@code n - 1} edges the clustering actually uses. */
    public boolean isSpanningTreeEdge(int pairIndex) {
        return this.edgeOfPair[pairIndex] >= 0;
    }

    /** Pair index of the given spanning tree edge, for {@code edge} in {@code [0, n-1)}. */
    public int getSpanningTreeEdgePair(int edge) {
        return pairIndex(this.edgeFirst[edge], this.edgeSecond[edge]);
    }

    public double getDistance(int pairIndex) {
        return this.distances[pairIndex];
    }

    public double getDistance(int i, int j) {
        return this.distances[pairIndex(i, j)];
    }

    /** Left child cluster of the {@code k}-th merge, merges being ordered by ascending height. */
    public int getMergeLeft(int merge) {
        return this.mergeLeft[merge];
    }

    public int getMergeRight(int merge) {
        return this.mergeRight[merge];
    }

    public double getMergeHeight(int merge) {
        return this.mergeHeight[merge];
    }

    /** Left child clusters of all merges, in ascending height order. Live view, do not modify. */
    public int[] getMergeLeftClusters() {
        return this.mergeLeft;
    }

    /** Right child clusters of all merges, in ascending height order. Live view, do not modify. */
    public int[] getMergeRightClusters() {
        return this.mergeRight;
    }

    /** Heights of all merges, in ascending order. Live view, do not modify. */
    public double[] getMergeHeights() {
        return this.mergeHeight;
    }

    public int getClusterSize(int cluster) {
        return this.clusterSize[cluster];
    }

    /**
     * Number of taxon pairs whose most recent common ancestor is the cluster created by the
     * {@code k}-th merge. These counts partition all {@code n(n-1)/2} pairs across the merges.
     */
    public int getCrossPairCount(int merge) {
        return this.clusterSize[this.mergeLeft[merge]] * this.clusterSize[this.mergeRight[merge]];
    }

    /**
     * Sets the distance of the pair {@code (i, j)} and repairs the spanning tree and the dendrogram.
     *
     * <p>Most changes do not need the dendrogram rebuilding at all. Only {@code n - 1} of the
     * {@code n(n-1)/2} entries are spanning tree edges, so an entry outside it usually leaves the
     * tree exactly as it was, and an entry inside it usually keeps its rank among the sorted weights,
     * which moves a single node height. See {@link UpdateKind}, and {@link #getLastChangedMerge()}
     * for which merge moved.
     */
    public UpdateKind update(int i, int j, double newDistance) {
        int pair = pairIndex(i, j);
        double oldDistance = this.distances[pair];
        this.distances[pair] = newDistance;

        int edge = this.edgeOfPair[pair];

        if (edge < 0) {
            // a non-spanning-tree edge can only enter the tree by becoming cheaper
            if (newDistance >= oldDistance || !insertIfCheaper(i, j, pair, newDistance)) {
                return UpdateKind.UNCHANGED;
            }

            rebuildDendrogram();
            return UpdateKind.STRUCTURE;
        }

        this.edgeWeight[edge] = newDistance;

        if (newDistance > oldDistance && repairAfterIncrease(edge)) {
            rebuildDendrogram();
            return UpdateKind.STRUCTURE;
        }

        // the edge is still in the spanning tree with the new weight. If it also keeps its rank, the
        // merges are unchanged and only the height of the one it created has moved
        int merge = this.rankOfEdge[edge];
        if (keepsItsRank(merge, newDistance)) {
            this.mergeHeight[merge] = newDistance / 2.0;
            this.lastChangedMerge = merge;
            return UpdateKind.HEIGHT_ONLY;
        }

        rebuildDendrogram();
        return UpdateKind.STRUCTURE;
    }

    /** Merge whose height moved, valid after {@link #update} returned {@link UpdateKind#HEIGHT_ONLY}. */
    public int getLastChangedMerge() {
        return this.lastChangedMerge;
    }

    private boolean keepsItsRank(int merge, double weight) {
        if (merge > 0 && weight < this.edgeWeight[this.edgeOrder[merge - 1]]) {
            return false;
        }

        return merge >= this.edgeOrder.length - 1
                || weight <= this.edgeWeight[this.edgeOrder[merge + 1]];
    }

    /**
     * Multiplies every distance by {@code factor}.
     *
     * <p>Scaling preserves the order of the entries, so Kruskal makes exactly the same choices: the
     * spanning tree, its edge set and the merge order all survive, and every node height is simply
     * multiplied through. That makes a scaling of the matrix precisely a scaling of the tree, with no
     * re-clustering.
     */
    public void scale(double factor) {
        for (int pair = 0; pair < this.pairCount; pair++) {
            this.distances[pair] *= factor;
        }
        for (int edge = 0; edge < this.edgeWeight.length; edge++) {
            this.edgeWeight[edge] *= factor;
        }
        for (int merge = 0; merge < this.mergeHeight.length; merge++) {
            this.mergeHeight[merge] *= factor;
        }
    }

    /** Commits the current state, discarding the rollback snapshot. */
    public void accept() {
        snapshot();
    }

    /** Rolls back to the state as of the last {@link #accept()} or construction. */
    public void restore() {
        if (this.storedWholeMatrix) {
            System.arraycopy(this.storedDistances, 0, this.distances, 0, this.pairCount);
            this.storedWholeMatrix = false;
        } else if (this.storedPairIndex >= 0) {
            this.distances[this.storedPairIndex] = this.storedDistance;
        }
        this.storedPairIndex = -1;

        System.arraycopy(this.storedEdgeFirst, 0, this.edgeFirst, 0, this.edgeFirst.length);
        System.arraycopy(this.storedEdgeSecond, 0, this.edgeSecond, 0, this.edgeSecond.length);
        System.arraycopy(this.storedEdgeWeight, 0, this.edgeWeight, 0, this.edgeWeight.length);
        System.arraycopy(this.storedEdgeOfPair, 0, this.edgeOfPair, 0, this.edgeOfPair.length);
        System.arraycopy(this.storedMergeLeft, 0, this.mergeLeft, 0, this.mergeLeft.length);
        System.arraycopy(this.storedMergeRight, 0, this.mergeRight, 0, this.mergeRight.length);
        System.arraycopy(this.storedMergeHeight, 0, this.mergeHeight, 0, this.mergeHeight.length);
        System.arraycopy(this.storedClusterSize, 0, this.clusterSize, 0, this.clusterSize.length);
        System.arraycopy(this.storedEdgeOrder, 0, this.edgeOrder, 0, this.edgeOrder.length);
        System.arraycopy(this.storedRankOfEdge, 0, this.rankOfEdge, 0, this.rankOfEdge.length);
    }

    /**
     * Records the pre-change value of a single entry so {@link #restore()} can undo it. Must be
     * called before {@link #update} and after the preceding {@link #accept()}.
     */
    public void storeDistance(int i, int j) {
        this.storedPairIndex = pairIndex(i, j);
        this.storedDistance = this.distances[this.storedPairIndex];
    }

    /**
     * Records the whole matrix so {@link #restore()} can undo a change that touches every entry,
     * such as {@link #scale}. Copying is exact, where undoing a scaling by multiplying through by its
     * reciprocal would leave rounding error behind on every rejected proposal.
     */
    public void storeMatrix() {
        System.arraycopy(this.distances, 0, this.storedDistances, 0, this.pairCount);
        this.storedWholeMatrix = true;
    }

    private void snapshot() {
        this.storedPairIndex = -1;
        this.storedWholeMatrix = false;
        System.arraycopy(this.edgeFirst, 0, this.storedEdgeFirst, 0, this.edgeFirst.length);
        System.arraycopy(this.edgeSecond, 0, this.storedEdgeSecond, 0, this.edgeSecond.length);
        System.arraycopy(this.edgeWeight, 0, this.storedEdgeWeight, 0, this.edgeWeight.length);
        System.arraycopy(this.edgeOfPair, 0, this.storedEdgeOfPair, 0, this.edgeOfPair.length);
        System.arraycopy(this.mergeLeft, 0, this.storedMergeLeft, 0, this.mergeLeft.length);
        System.arraycopy(this.mergeRight, 0, this.storedMergeRight, 0, this.mergeRight.length);
        System.arraycopy(this.mergeHeight, 0, this.storedMergeHeight, 0, this.mergeHeight.length);
        System.arraycopy(this.clusterSize, 0, this.storedClusterSize, 0, this.clusterSize.length);
        System.arraycopy(this.edgeOrder, 0, this.storedEdgeOrder, 0, this.edgeOrder.length);
        System.arraycopy(this.rankOfEdge, 0, this.storedRankOfEdge, 0, this.rankOfEdge.length);
    }

    /**
     * Builds the spanning tree from scratch with Prim's algorithm, which is O(n^2) on a complete
     * graph and so matches the size of the matrix itself.
     */
    private void buildMinimumSpanningTree() {
        Arrays.fill(this.edgeOfPair, -1);

        for (int edge = 0; edge < this.edgeOrder.length; edge++) {
            this.edgeOrder[edge] = edge;
        }

        double[] best = new double[this.leafCount];
        int[] bestFrom = new int[this.leafCount];
        boolean[] inTree = new boolean[this.leafCount];

        inTree[0] = true;
        for (int taxon = 1; taxon < this.leafCount; taxon++) {
            best[taxon] = this.distances[pairIndex(0, taxon)];
            bestFrom[taxon] = 0;
        }

        for (int edge = 0; edge < this.leafCount - 1; edge++) {
            int next = -1;
            for (int taxon = 1; taxon < this.leafCount; taxon++) {
                if (!inTree[taxon] && (next < 0 || best[taxon] < best[next])) {
                    next = taxon;
                }
            }

            setEdge(edge, bestFrom[next], next, best[next]);
            inTree[next] = true;

            for (int taxon = 1; taxon < this.leafCount; taxon++) {
                if (inTree[taxon]) {
                    continue;
                }

                double distance = this.distances[pairIndex(next, taxon)];
                if (distance < best[taxon]) {
                    best[taxon] = distance;
                    bestFrom[taxon] = next;
                }
            }
        }
    }

    private void setEdge(int edge, int first, int second, double weight) {
        this.edgeFirst[edge] = first;
        this.edgeSecond[edge] = second;
        this.edgeWeight[edge] = weight;
        this.edgeOfPair[pairIndex(first, second)] = edge;
    }

    /**
     * Swaps the pair {@code (i, j)} into the spanning tree if it is cheaper than the heaviest edge
     * on the tree path between {@code i} and {@code j}.
     */
    private boolean insertIfCheaper(int i, int j, int pair, double weight) {
        buildAdjacency();

        // walk the spanning tree from i, tracking the heaviest edge seen on the way to each taxon
        Arrays.fill(this.visited, false);
        this.visited[i] = true;
        this.maxEdgeOnPath[i] = -1;
        this.queue[0] = i;

        for (int head = 0, tail = 1; head < tail; head++) {
            int node = this.queue[head];
            for (int slot = this.adjacencyStart[node]; slot < this.adjacencyStart[node + 1]; slot++) {
                int neighbour = this.adjacencyNode[slot];
                if (this.visited[neighbour]) {
                    continue;
                }

                int edge = this.adjacencyEdge[slot];
                int heaviest = this.maxEdgeOnPath[node];
                if (heaviest < 0 || this.edgeWeight[edge] > this.edgeWeight[heaviest]) {
                    heaviest = edge;
                }

                this.visited[neighbour] = true;
                this.maxEdgeOnPath[neighbour] = heaviest;
                this.queue[tail++] = neighbour;
            }
        }

        int heaviest = this.maxEdgeOnPath[j];
        if (weight >= this.edgeWeight[heaviest]) {
            return false;
        }

        this.edgeOfPair[pairIndex(this.edgeFirst[heaviest], this.edgeSecond[heaviest])] = -1;
        this.edgeFirst[heaviest] = Math.min(i, j);
        this.edgeSecond[heaviest] = Math.max(i, j);
        this.edgeWeight[heaviest] = weight;
        this.edgeOfPair[pair] = heaviest;
        return true;
    }

    /**
     * Repairs the spanning tree after a tree edge got more expensive: cutting it splits the taxa in
     * two, and the cheapest pair crossing that cut becomes the replacement. The crossing pairs are
     * exactly the pairs whose most recent common ancestor is the node this edge creates.
     */
    private boolean repairAfterIncrease(int edge) {
        int from = this.edgeFirst[edge];
        int to = this.edgeSecond[edge];

        buildAdjacencyWithout(edge);

        Arrays.fill(this.visited, false);
        this.visited[from] = true;
        this.queue[0] = from;
        int size = 1;

        for (int head = 0; head < size; head++) {
            int node = this.queue[head];
            for (int slot = this.adjacencyStart[node]; slot < this.adjacencyStart[node + 1]; slot++) {
                int neighbour = this.adjacencyNode[slot];
                if (!this.visited[neighbour]) {
                    this.visited[neighbour] = true;
                    this.queue[size++] = neighbour;
                }
            }
        }

        int bestFirst = from;
        int bestSecond = to;
        double bestWeight = this.edgeWeight[edge];

        for (int head = 0; head < size; head++) {
            int inside = this.queue[head];
            for (int outside = 0; outside < this.leafCount; outside++) {
                if (this.visited[outside]) {
                    continue;
                }

                double weight = this.distances[pairIndex(inside, outside)];
                if (weight < bestWeight) {
                    bestWeight = weight;
                    bestFirst = inside;
                    bestSecond = outside;
                }
            }
        }

        if (bestFirst == from && bestSecond == to) {
            return false;
        }

        this.edgeOfPair[pairIndex(from, to)] = -1;
        setEdge(edge, Math.min(bestFirst, bestSecond), Math.max(bestFirst, bestSecond), bestWeight);
        return true;
    }

    private void buildAdjacency() {
        buildAdjacencyWithout(-1);
    }

    private void buildAdjacencyWithout(int excludedEdge) {
        Arrays.fill(this.adjacencyStart, 0);

        for (int edge = 0; edge < this.edgeFirst.length; edge++) {
            if (edge == excludedEdge) {
                continue;
            }
            this.adjacencyStart[this.edgeFirst[edge] + 1]++;
            this.adjacencyStart[this.edgeSecond[edge] + 1]++;
        }

        for (int taxon = 0; taxon < this.leafCount; taxon++) {
            this.adjacencyStart[taxon + 1] += this.adjacencyStart[taxon];
        }

        System.arraycopy(this.adjacencyStart, 0, this.parent, 0, this.leafCount);

        for (int edge = 0; edge < this.edgeFirst.length; edge++) {
            if (edge == excludedEdge) {
                continue;
            }

            int first = this.edgeFirst[edge];
            int second = this.edgeSecond[edge];

            this.adjacencyNode[this.parent[first]] = second;
            this.adjacencyEdge[this.parent[first]++] = edge;
            this.adjacencyNode[this.parent[second]] = first;
            this.adjacencyEdge[this.parent[second]++] = edge;
        }
    }

    /**
     * Replays Kruskal over the spanning tree edges in ascending weight order. Ascending order makes
     * merge heights automatically increasing, so the dendrogram is always a valid ultrametric tree.
     */
    private void rebuildDendrogram() {
        sortEdgesByWeight();

        for (int taxon = 0; taxon < this.leafCount; taxon++) {
            this.parent[taxon] = taxon;
            this.clusterOf[taxon] = taxon;
            this.clusterSize[taxon] = 1;
        }

        for (int merge = 0; merge < this.edgeOrder.length; merge++) {
            int edge = this.edgeOrder[merge];
            this.rankOfEdge[edge] = merge;

            int first = find(this.edgeFirst[edge]);
            int second = find(this.edgeSecond[edge]);

            int left = this.clusterOf[first];
            int right = this.clusterOf[second];
            int cluster = this.leafCount + merge;

            this.mergeLeft[merge] = left;
            this.mergeRight[merge] = right;
            this.mergeHeight[merge] = this.edgeWeight[edge] / 2.0;
            this.clusterSize[cluster] = this.clusterSize[left] + this.clusterSize[right];

            this.parent[second] = first;
            this.clusterOf[first] = cluster;
        }
    }

    /**
     * Insertion sort over the spanning tree edges. Only one weight moves per update, so the array is
     * already almost sorted and this runs in linear time, without the boxing an {@code Integer[]}
     * comparator sort would need.
     */
    private void sortEdgesByWeight() {
        for (int position = 1; position < this.edgeOrder.length; position++) {
            int edge = this.edgeOrder[position];
            double weight = this.edgeWeight[edge];

            int hole = position;
            while (hole > 0 && this.edgeWeight[this.edgeOrder[hole - 1]] > weight) {
                this.edgeOrder[hole] = this.edgeOrder[hole - 1];
                hole--;
            }

            this.edgeOrder[hole] = edge;
        }
    }

    private int find(int taxon) {
        int root = taxon;
        while (this.parent[root] != root) {
            root = this.parent[root];
        }

        while (this.parent[taxon] != root) {
            int next = this.parent[taxon];
            this.parent[taxon] = root;
            taxon = next;
        }

        return root;
    }
}
