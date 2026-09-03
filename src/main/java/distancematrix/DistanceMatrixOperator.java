package distancematrix;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.List;

/**
 * Samples an ultrametric tree by carrying a full taxon-by-taxon distance matrix as augmented state.
 *
 * <p>The tree is at all times the single-linkage tree of the matrix, that is, the tree obtained by
 * walking the taxon pairs in ascending order of distance and merging whenever the endpoints are not
 * yet connected. A proposal perturbs one matrix entry by a Gaussian random walk whose scale is
 * adapted by Robbins-Monro, and the tree is repaired incrementally, keeping node identities
 * wherever the topology did not move.
 *
 * <p>Single linkage is what makes this work on a time tree. It is ultrametric-valued for any input
 * matrix, so a perturbed matrix always maps onto a valid rooted ultrametric tree with no rooting or
 * tip-repair step, and it reproduces an ultrametric matrix exactly, so the matrix can be
 * initialised from the state tree's own distances without overwriting the tree.
 *
 * <h2>Hastings ratio</h2>
 *
 * <p>The walk on a single entry is symmetric, so the Hastings factor of the walk itself is one.
 * That alone is not enough: it makes the chain reversible with respect to {@code pi(SL(D))} on
 * matrix space, but the matrix has {@code n(n-1)/2} entries against the tree's {@code n - 1}
 * heights, so the map is many-to-one, its fibres are unbounded, and the tree marginal would come out
 * as {@code pi(T) * V(T)}.
 *
 * <p>Single linkage makes {@code V(T)} computable in closed form. Only the {@code n - 1} minimum
 * spanning tree edges of the matrix are ever used, every taxon pair belongs to exactly one internal
 * node, and the fibre factorises over those groups, so the fibre volume is a product of {@code n-1}
 * group-minimum densities under the reference density placed on the entries. See
 * {@link ReferenceDensity#logFibreVolume}. Dividing the target by it makes the tree marginal exactly
 * {@code pi(T)}, and the proposal returns
 *
 * <pre>
 *     log g(D') - log g(D) + log V(T) - log V(T')
 * </pre>
 *
 * <p>The fibre term drops out whenever the tree is unchanged, leaving a plain Metropolis step on the
 * chosen entry under its reference density. That is the common case, since a uniformly chosen entry
 * is a spanning tree edge only with probability {@code 2 / n}.
 *
 * <h2>Working alongside other operators</h2>
 *
 * <p>The matrix is private to this operator, so anything else that changes the tree breaks the
 * invariant. Two things do: BEAST's state node initialisers, a {@code RandomTree} among them, which
 * run from inside {@code MCMC.run()} long after every {@code initAndValidate()}, and any other
 * operator acting on the same tree. Every proposal therefore starts by checking that the tree is
 * still the clustering of the matrix, and rebuilds the matrix from the tree when it is not. That
 * rebuild draws from the exact conditional of the matrix given the tree, so it is a Gibbs step and
 * costs nothing in correctness, which is what makes this operator safe to mix with any other tree
 * operator.
 *
 * <h2>Limitations</h2>
 *
 * <p>Tips are assumed contemporaneous and at height zero. Serially sampled trees are rejected.
 *
 * <p>Node heights high in the tree move very slowly. Single linkage sets a node's distance to the
 * minimum over the {@code m_v} pairs crossing it, and a proposal moves one entry, so raising a node
 * near the root means raising all of its {@code m_v} entries one at a time. At sixty taxa the root
 * has roughly nine hundred crossing pairs, and in practice the overall tree scale barely moves. Pair
 * this operator with a tree scaling operator rather than running it alone.
 */
public class DistanceMatrixOperator extends TreeOperator {

    public final Input<String> referenceDensityInput = new Input<>(
            "referenceDensity",
            "reference density placed on every matrix entry, either exponential or lognormal",
            "lognormal");

    public final Input<Double> referenceMeanInput = new Input<>(
            "referenceMean",
            "mean of the reference density; defaults to the mean of the initial distances",
            Double.NaN);

    public final Input<Double> referenceShapeInput = new Input<>(
            "referenceShape",
            "log standard deviation of the lognormal reference density",
            1.0);

    public final Input<Double> scaleFactorInput = new Input<>(
            "scaleFactor",
            "initial standard deviation of the random walk on a matrix entry; "
                    + "defaults to a tenth of the mean of the initial distances",
            Double.NaN);

    public final Input<Double> spanningTreeProbabilityInput = new Input<>(
            "spanningTreeProbability",
            "probability of perturbing one of the n-1 entries the clustering actually uses, rather "
                    + "than one of the others. Choosing uniformly would spend a fraction 1 - 2/n of "
                    + "the proposals on entries the tree does not depend on, which is 97 per cent at "
                    + "sixty taxa",
            0.5);

    public final Input<Double> scaleProbabilityInput = new Input<>(
            "scaleProbability",
            "probability of scaling the whole matrix instead of perturbing a single entry. Single "
                    + "linkage reads a node distance as the minimum over the pairs crossing it, so "
                    + "lifting a node near the root one entry at a time is hopeless and the overall "
                    + "tree scale barely moves without this",
            0.05);

    public final Input<Boolean> correctFibreVolumeInput = new Input<>(
            "correctFibreVolume",
            "whether to divide out the fibre volume of the matrix-to-tree map. Turning this off "
                    + "leaves a plain symmetric walk that returns a Hastings ratio of one, which "
                    + "does not target the tree posterior and is only useful for comparison runs",
            true);

    private static final double HEIGHT_TOLERANCE = 1e-10;

    // the scale move cannot use calcDelta, since the schedule allows only one coercible parameter
    // and that is the random walk width, so it adapts on its own decaying schedule as in
    // irreversible.GuidedRandomWalkOperator
    private static final double ADAPTATION_OFFSET = 10.0;
    private static final double ADAPTATION_EXPONENT = 0.6;
    private static final double SCALE_TARGET_ACCEPTANCE = 0.3;
    private static final double MIN_SCALE_WINDOW = 1e-8;
    private static final double MAX_SCALE_WINDOW = 10.0;

    /** Which move the last proposal made, so that {@link #optimize} adapts the right width. */
    private enum MoveKind { NONE, WALK, SCALE }

    private Tree tree;
    private SingleLinkageTree clustering;
    private ReferenceDensity referenceDensity;
    private boolean correctFibreVolume;
    private double spanningTreeProbability;

    private int[] pairFirst;
    private int[] pairSecond;
    private double[] treeDistances;

    private Node[] nodeOfCluster;

    // what the tree looked like when this operator last agreed with it, by node number. Node
    // numbering survives BEAST's tree store and restore, which swap the two node arrays but keep
    // every number, so this stays a valid index.
    private double[] expectedHeight;
    private int[] expectedLeft;
    private int[] expectedRight;
    private int expectedRoot;
    private double[] storedExpectedHeight;
    private int[] storedExpectedLeft;
    private int[] storedExpectedRight;
    private int storedExpectedRoot;

    private double scaleProbability;
    private double scaleWindow = 0.1;
    private long scaleWindowCount = 0;
    private MoveKind lastMove = MoveKind.NONE;

    private double scaleFactor;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.correctFibreVolume = this.correctFibreVolumeInput.get();
        this.spanningTreeProbability = this.spanningTreeProbabilityInput.get();
        this.scaleProbability = this.scaleProbabilityInput.get();

        if (!(this.scaleProbability >= 0.0) || !(this.scaleProbability < 1.0)) {
            throw new IllegalArgumentException("scaleProbability must lie in [0, 1)");
        }

        if (!(this.spanningTreeProbability > 0.0) || !(this.spanningTreeProbability < 1.0)) {
            throw new IllegalArgumentException("spanningTreeProbability must lie strictly between "
                    + "zero and one, so that every entry stays reachable in both directions");
        }

        int leafCount = this.tree.getLeafNodeCount();
        if (leafCount < 3) {
            throw new IllegalArgumentException("DistanceMatrixOperator requires at least three taxa");
        }

        int pairCount = leafCount * (leafCount - 1) / 2;
        this.pairFirst = new int[pairCount];
        this.pairSecond = new int[pairCount];
        this.treeDistances = new double[pairCount];

        int nodeCount = 2 * leafCount - 1;
        this.nodeOfCluster = new Node[nodeCount];
        this.expectedHeight = new double[nodeCount];
        this.expectedLeft = new int[nodeCount];
        this.expectedRight = new int[nodeCount];
        this.storedExpectedHeight = new double[nodeCount];
        this.storedExpectedLeft = new int[nodeCount];
        this.storedExpectedRight = new int[nodeCount];

        int pair = 0;
        for (int first = 0; first < leafCount; first++) {
            for (int second = first + 1; second < leafCount; second++) {
                this.pairFirst[pair] = first;
                this.pairSecond[pair] = second;
                pair++;
            }
        }

        this.scaleFactor = Double.isNaN(this.scaleFactorInput.get())
                ? 1.0
                : this.scaleFactorInput.get();

        // the matrix itself is built on the first proposal rather than here. BEAST runs its state
        // node initialisers, a RandomTree among them, from inside MCMC.run(), which is long after
        // every initAndValidate() call, so the tree available at this point is not the tree the
        // chain will actually start from.
    }

    @Override
    public double proposal() {
        this.lastMove = MoveKind.NONE;
        synchroniseWithTree();

        if (this.correctFibreVolume && Randomizer.nextDouble() < this.scaleProbability) {
            this.lastMove = MoveKind.SCALE;
            return scaleProposal();
        }

        this.lastMove = MoveKind.WALK;
        return walkProposal();
    }

    /**
     * Scales every matrix entry, and with it the whole tree.
     *
     * <p>Single linkage is scale equivariant: multiplying the matrix through preserves the order of
     * the entries, so Kruskal makes identical choices and the tree comes out with the same topology
     * and every height multiplied. One scaling of the matrix is exactly one scaling of the tree.
     *
     * <p>The scale is drawn log-symmetrically, which is what BEAST's newer scale operators do, so the
     * proposal density cancels and the only kernel contribution is the Jacobian of scaling the
     * {@code n(n-1)/2} entries.
     */
    private double scaleProposal() {
        double logScale = this.scaleWindow * Randomizer.nextGaussian();
        double scale = Math.exp(logScale);

        if (!(scale > 0.0) || !Double.isFinite(scale)) {
            return Double.NEGATIVE_INFINITY;
        }

        double oldLogFibreVolume = this.referenceDensity.logFibreVolume(this.clustering);
        double oldLogDensity = totalLogReferenceDensity();

        this.clustering.storeMatrix();
        this.clustering.scale(scale);

        this.tree.startEditing(this);
        writeMergeHeights();

        double logRatio = this.clustering.getPairCount() * logScale
                + totalLogReferenceDensity() - oldLogDensity
                + oldLogFibreVolume - this.referenceDensity.logFibreVolume(this.clustering);

        return Double.isFinite(logRatio) ? logRatio : Double.NEGATIVE_INFINITY;
    }

    private double walkProposal() {
        int pair = choosePair();
        int first = this.pairFirst[pair];
        int second = this.pairSecond[pair];

        double logSelectionBefore = logSelectionProbability(pair);

        double oldDistance = this.clustering.getDistance(pair);
        double newDistance = oldDistance + this.scaleFactor * Randomizer.nextGaussian();

        if (!(newDistance > 0.0)) {
            return Double.NEGATIVE_INFINITY;
        }

        double oldLogFibreVolume = this.correctFibreVolume
                ? this.referenceDensity.logFibreVolume(this.clustering)
                : 0.0;

        this.clustering.storeDistance(first, second);
        SingleLinkageTree.UpdateKind kind = this.clustering.update(first, second, newDistance);

        applyUpdate(kind);

        if (!this.correctFibreVolume) {
            return 0.0;
        }

        double logRatio = this.referenceDensity.logDensity(newDistance)
                - this.referenceDensity.logDensity(oldDistance)
                + logSelectionProbability(pair) - logSelectionBefore;

        if (kind != SingleLinkageTree.UpdateKind.UNCHANGED) {
            logRatio += oldLogFibreVolume - this.referenceDensity.logFibreVolume(this.clustering);
        }

        return Double.isFinite(logRatio) ? logRatio : Double.NEGATIVE_INFINITY;
    }

    /**
     * Carries a clustering update over to the tree, touching as little of it as possible.
     *
     * <p>A node marked dirty costs the tree likelihood a recomputation of its partials and of every
     * node above it, so installing the whole topology for what is usually a single height change is
     * the difference between a local update and a full traversal.
     */
    private void applyUpdate(SingleLinkageTree.UpdateKind kind) {
        switch (kind) {
            case UNCHANGED -> {
            }
            case HEIGHT_ONLY -> {
                int merge = this.clustering.getLastChangedMerge();
                Node node = this.nodeOfCluster[this.tree.getLeafNodeCount() + merge];

                this.tree.startEditing(this);
                node.setHeight(this.clustering.getMergeHeight(merge));
                this.expectedHeight[node.getNr()] = node.getHeight();
            }
            case STRUCTURE -> {
                this.tree.startEditing(this);
                TreeUtils.applyClustering(
                        this.tree,
                        this.clustering.getMergeLeftClusters(),
                        this.clustering.getMergeRightClusters(),
                        this.clustering.getMergeHeights(),
                        this.nodeOfCluster);
                cacheTree();
            }
        }
    }

    /** Writes every merge height onto its node, used after the whole matrix was scaled. */
    private void writeMergeHeights() {
        int leafCount = this.tree.getLeafNodeCount();

        for (int merge = 0; merge < leafCount - 1; merge++) {
            Node node = this.nodeOfCluster[leafCount + merge];
            node.setHeight(this.clustering.getMergeHeight(merge));
            this.expectedHeight[node.getNr()] = node.getHeight();
        }
    }

    private double totalLogReferenceDensity() {
        double total = 0.0;

        for (int pair = 0; pair < this.clustering.getPairCount(); pair++) {
            total += this.referenceDensity.logDensity(this.clustering.getDistance(pair));
        }

        return total;
    }

    @Override
    public void accept() {
        super.accept();

        if (this.clustering != null) {
            this.clustering.accept();
            storeCache();
        }
    }

    @Override
    public void reject(int reason) {
        super.reject(reason);

        if (this.clustering != null) {
            // MCMC rejects the operator before restoring the state, so the cache cannot be re-read
            // from the tree here and has to be rolled back alongside the matrix
            this.clustering.restore();
            restoreCache();
        }
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    @Override
    public double getCoercableParameterValue() {
        return this.scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.scaleFactor = value;
    }

    @Override
    public void optimize(double logAlpha) {
        if (this.lastMove == MoveKind.SCALE) {
            optimiseScaleWindow(logAlpha);
            return;
        }

        if (this.lastMove != MoveKind.WALK) {
            return;
        }

        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.scaleFactor);
        this.scaleFactor = Math.exp(delta);
    }

    /**
     * Robbins-Monro on the scale move's own width and its own counter. It cannot go through
     * {@code calcDelta}, because the schedule reads {@code getCoercableParameterValue} and so tunes
     * one parameter per operator, which here is the random walk width.
     */
    private void optimiseScaleWindow(double logAlpha) {
        this.scaleWindowCount++;

        double alpha = Math.exp(Math.min(logAlpha, 0.0));
        if (!Double.isFinite(alpha)) {
            return;
        }

        double learningRate =
                Math.pow(this.scaleWindowCount + ADAPTATION_OFFSET, -ADAPTATION_EXPONENT);
        double window = Math.exp(Math.log(this.scaleWindow)
                + learningRate * (alpha - SCALE_TARGET_ACCEPTANCE));

        this.scaleWindow = Math.min(MAX_SCALE_WINDOW, Math.max(MIN_SCALE_WINDOW, window));
    }

    /**
     * Picks the entry to perturb, favouring the entries the clustering uses.
     *
     * <p>Only the {@code n - 1} spanning tree edges carry the tree: each one is a node height, and
     * every other entry leaves the tree exactly as it was. Choosing uniformly would therefore waste
     * a fraction {@code 1 - 2/n} of the proposals, and would reach any particular node height only
     * once in {@code n(n-1)/2} tries. The entries outside the spanning tree still have to move,
     * since that is how the topology changes, so both groups keep a positive probability.
     */
    private int choosePair() {
        if (!this.correctFibreVolume) {
            // the uncorrected mode is meant to be the plain symmetric walk, and a biased choice of
            // entry is not symmetric
            return Randomizer.nextInt(this.clustering.getPairCount());
        }

        int edgeCount = this.clustering.getLeafCount() - 1;

        if (Randomizer.nextDouble() < this.spanningTreeProbability) {
            return this.clustering.getSpanningTreeEdgePair(Randomizer.nextInt(edgeCount));
        }

        int offset = Randomizer.nextInt(this.clustering.getPairCount() - edgeCount);
        for (int pair = 0; pair < this.clustering.getPairCount(); pair++) {
            if (!this.clustering.isSpanningTreeEdge(pair) && offset-- == 0) {
                return pair;
            }
        }

        throw new IllegalStateException("the spanning tree does not have " + edgeCount + " edges");
    }

    /**
     * Log probability that {@link #choosePair()} picks this entry, given the current matrix. The
     * move is symmetric in the perturbation but not in the choice of entry, because an entry can
     * join or leave the spanning tree, so this enters the Hastings ratio as the reverse minus the
     * forward probability.
     */
    private double logSelectionProbability(int pair) {
        if (!this.correctFibreVolume) {
            return 0.0;
        }

        int edgeCount = this.clustering.getLeafCount() - 1;

        if (this.clustering.isSpanningTreeEdge(pair)) {
            return Math.log(this.spanningTreeProbability) - Math.log(edgeCount);
        }

        return Math.log(1.0 - this.spanningTreeProbability)
                - Math.log(this.clustering.getPairCount() - edgeCount);
    }

    /**
     * Makes sure the matrix still induces the tree, rebuilding it from the tree when it does not.
     *
     * <p>The operator's whole contract is that the tree is the single-linkage tree of the matrix, and
     * the matrix is private to the operator, so anything that changes the tree from the outside
     * silently breaks that. Two things do: BEAST's state node initialisers, which run from inside
     * {@code MCMC.run()} well after every {@code initAndValidate()}, and any other operator acting on
     * the same tree. Left unchecked, the next proposal that touches the tree overwrites it with the
     * tree of a stale matrix, which pins the tree in place for the rest of the run.
     *
     * <p>Rebuilding is an exact Gibbs step, so it costs nothing in correctness. Conditioned on the
     * tree, the target leaves the matrix entries independent across the groups of pairs sharing a
     * most recent common ancestor, and within a group they are draws from the reference density
     * conditioned on all exceeding the node distance with the minimum attaining it. Sampling that is
     * a matter of picking which entry attains the minimum and drawing the rest from the tail.
     */
    private void synchroniseWithTree() {
        if (this.clustering != null && treeMatchesCache()) {
            return;
        }

        checkTipsAreContemporaneous();
        writeTreeDistances();

        if (this.referenceDensity == null) {
            double sum = 0.0;
            for (double distance : this.treeDistances) {
                sum += distance;
            }

            double meanDistance = sum / this.treeDistances.length;
            if (!(meanDistance > 0.0) || !Double.isFinite(meanDistance)) {
                throw new IllegalArgumentException("the tree has no positive taxon distances");
            }

            this.referenceDensity = createReferenceDensity(meanDistance);

            if (Double.isNaN(this.scaleFactorInput.get())) {
                this.scaleFactor = meanDistance / 10.0;
            }
        }

        double[] distances = sampleMatrixGivenTree();

        if (this.clustering == null) {
            this.clustering = new SingleLinkageTree(this.tree.getLeafNodeCount(), distances);
        } else {
            this.clustering.reset(distances);
        }

        // the matrix was drawn from its conditional given the tree, so the clustering already is the
        // tree. Installing it is therefore a no-op that leaves every node clean, and it hands back
        // the cluster to node mapping the height fast path needs.
        TreeUtils.applyClustering(
                this.tree,
                this.clustering.getMergeLeftClusters(),
                this.clustering.getMergeRightClusters(),
                this.clustering.getMergeHeights(),
                this.nodeOfCluster);

        cacheTree();
        storeCache();
    }

    /**
     * Whether the tree still looks the way this operator left it, in time proportional to the number
     * of nodes and without allocating.
     *
     * <p>The cache holds exactly the values this operator wrote, so the heights compare exactly. A
     * cheap check matters because it runs on every proposal: a stale matrix means a wrong Hastings
     * ratio, so skipping the check some of the time would be a correctness bug rather than a saving.
     */
    private boolean treeMatchesCache() {
        if (this.tree.getRoot().getNr() != this.expectedRoot) {
            return false;
        }

        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);

            if (node.getHeight() != this.expectedHeight[nodeNr]) {
                return false;
            }

            if (node.isLeaf()) {
                if (this.expectedLeft[nodeNr] >= 0) {
                    return false;
                }
                continue;
            }

            if (node.getChildCount() != 2
                    || node.getChild(0).getNr() != this.expectedLeft[nodeNr]
                    || node.getChild(1).getNr() != this.expectedRight[nodeNr]) {
                return false;
            }
        }

        return true;
    }

    private void cacheTree() {
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            this.expectedHeight[nodeNr] = node.getHeight();

            if (node.isLeaf()) {
                this.expectedLeft[nodeNr] = -1;
                this.expectedRight[nodeNr] = -1;
            } else {
                this.expectedLeft[nodeNr] = node.getChild(0).getNr();
                this.expectedRight[nodeNr] = node.getChild(1).getNr();
            }
        }

        this.expectedRoot = this.tree.getRoot().getNr();
    }

    private void storeCache() {
        System.arraycopy(this.expectedHeight, 0, this.storedExpectedHeight, 0, this.expectedHeight.length);
        System.arraycopy(this.expectedLeft, 0, this.storedExpectedLeft, 0, this.expectedLeft.length);
        System.arraycopy(this.expectedRight, 0, this.storedExpectedRight, 0, this.expectedRight.length);
        this.storedExpectedRoot = this.expectedRoot;
    }

    private void restoreCache() {
        System.arraycopy(this.storedExpectedHeight, 0, this.expectedHeight, 0, this.expectedHeight.length);
        System.arraycopy(this.storedExpectedLeft, 0, this.expectedLeft, 0, this.expectedLeft.length);
        System.arraycopy(this.storedExpectedRight, 0, this.expectedRight, 0, this.expectedRight.length);
        this.expectedRoot = this.storedExpectedRoot;
    }

    /**
     * Draws the matrix from its exact conditional given the tree: within each group of pairs sharing
     * a most recent common ancestor, one entry attains the node distance and the rest sit above it.
     */
    private double[] sampleMatrixGivenTree() {
        double[] distances = new double[this.treeDistances.length];
        sampleMatrixBelow(this.tree.getRoot(), distances);
        return distances;
    }

    private int[] sampleMatrixBelow(Node node, double[] distances) {
        if (node.isLeaf()) {
            return new int[]{node.getNr()};
        }

        int[] left = sampleMatrixBelow(node.getChild(0), distances);
        int[] right = sampleMatrixBelow(node.getChild(1), distances);

        double nodeDistance = 2.0 * node.getHeight();
        int minimumIndex = Randomizer.nextInt(left.length * right.length);
        int index = 0;

        for (int a : left) {
            for (int b : right) {
                distances[this.clusteringPairIndex(a, b)] = index == minimumIndex
                        ? nodeDistance
                        : this.referenceDensity.sampleAbove(nodeDistance);
                index++;
            }
        }

        int[] clade = new int[left.length + right.length];
        System.arraycopy(left, 0, clade, 0, left.length);
        System.arraycopy(right, 0, clade, left.length, right.length);
        return clade;
    }

    private void writeTreeDistances() {
        writeTreeDistancesBelow(this.tree.getRoot(), this.treeDistances);
    }

    private int[] writeTreeDistancesBelow(Node node, double[] distances) {
        if (node.isLeaf()) {
            return new int[]{node.getNr()};
        }

        int[] left = writeTreeDistancesBelow(node.getChild(0), distances);
        int[] right = writeTreeDistancesBelow(node.getChild(1), distances);

        double nodeDistance = 2.0 * node.getHeight();
        for (int a : left) {
            for (int b : right) {
                distances[this.clusteringPairIndex(a, b)] = nodeDistance;
            }
        }

        int[] clade = new int[left.length + right.length];
        System.arraycopy(left, 0, clade, 0, left.length);
        System.arraycopy(right, 0, clade, left.length, right.length);
        return clade;
    }

    private int clusteringPairIndex(int first, int second) {
        int leafCount = this.tree.getLeafNodeCount();
        int smaller = Math.min(first, second);
        int larger = Math.max(first, second);
        return smaller * leafCount - smaller * (smaller + 1) / 2 + (larger - smaller - 1);
    }

    private ReferenceDensity createReferenceDensity(double meanDistance) {
        double mean = Double.isNaN(this.referenceMeanInput.get())
                ? meanDistance
                : this.referenceMeanInput.get();

        return switch (this.referenceDensityInput.get()) {
            case "exponential" -> new ExponentialReferenceDensity(1.0 / mean);
            case "lognormal" -> new LogNormalReferenceDensity(
                    Math.log(mean), this.referenceShapeInput.get());
            default -> throw new IllegalArgumentException(
                    "unknown reference density " + this.referenceDensityInput.get()
                            + ", expected exponential or lognormal");
        };
    }

    /**
     * The clustering reads a node distance as twice its height, so it places the tips at height
     * zero. Anything else would be silently shifted, hence the hard check.
     */
    private void checkTipsAreContemporaneous() {
        for (int leaf = 0; leaf < this.tree.getLeafNodeCount(); leaf++) {
            Node node = this.tree.getNode(leaf);

            if (Math.abs(node.getHeight()) > HEIGHT_TOLERANCE) {
                throw new IllegalArgumentException(
                        "DistanceMatrixOperator assumes ultrametricity with tips at height zero, "
                                + "but taxon " + node.getID() + " sits at height "
                                + node.getHeight());
            }
        }
    }
}
