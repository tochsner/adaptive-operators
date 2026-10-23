package transport;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.State;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import slice.SliceOperator;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class GuidedNodeTransportOperator extends SliceOperator {

    public Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    public final Input<String> weightingSchemeInput = new Input<>("weightingScheme", "", "inverse"
    );
    public final Input<Boolean> sampleCurrentBranchInput = new Input<>("sampleCurrentBranch",
            "sample the attachment from the whole path, including the branch the node already hangs on. "
                    + "Off by default, so that every proposal changes the topology and moving the node "
                    + "within its branch is left to the cheaper height operators", false);
    private static final double MIN_DISTANCE = 1.0E-12;

    int BURN_IN = 0;

    Tree tree;
    Alignment alignment;
    private boolean debug = false;
    FastJCDistance distance;
    WeightingScheme weightingScheme;
    boolean sampleCurrentBranch;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();

        this.distance = new FastJCDistance();
        this.distance.setPatterns(this.alignment);
        this.weightingScheme = WeightingScheme.fromName(this.weightingSchemeInput.get());
        this.sampleCurrentBranch = this.sampleCurrentBranchInput.get();
    }

    int count = 0;

    // numbers of the nodes the current proposal can leave stale in the likelihood cache, null if unknown
    private int[] affectedNodeNrs = null;

    @Override
    public double proposal() {
        throw new UnsupportedOperationException();
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood, State state) {
        this.affectedNodeNrs = null;
        this.count++;
        if (this.count < BURN_IN) return Double.NEGATIVE_INFINITY;
        else if (this.count == BURN_IN) System.out.println("Start transport");

        // select the first reference at random

        int referenceId1 = Randomizer.nextInt(this.tree.getLeafNodeCount());
        Node reference1 = this.tree.getNode(referenceId1);

        // select the second reference such that it is not the direct sibling

        List<Node> reference2Candidates = this.getReference2Candidates(reference1);
        Node reference2 = reference2Candidates.get(Randomizer.nextInt(reference2Candidates.size()));

        // select the node to move on the path between the two references in a weighted manner

        WeightedCandidates nodeCandidates = this.getWeightedNodeCandidates(reference1, reference2);
        Node node = nodeCandidates.candidates().get(this.sampleWeightedIndex(nodeCandidates.weights()));

        // record where the node hangs before the tree changes, a rejection has to recalculate this spot

        Node originalSibling = TreeUtils.getOtherChild(node.getParent(), node);

        // compute the old triplet selection density
        // (reuses the lists above instead of recomputing them: the tree has not been mutated yet,
        // so getTripletLogProbability would recompute the exact same candidates and weights)

        double forwardHR = this.getTripletLogProbability(reference2Candidates, reference2, nodeCandidates, node);

        // get the current and max distance

        double currentDistance = TreeUtils.getDistance(reference1, node.getParent());
        double maxDistance = TreeUtils.getDistance(reference1, reference2);
        double minAttachmentDistance = Math.max(0.0, node.getHeight() - reference1.getHeight());
        double maxAttachmentDistance = maxDistance - Math.max(0.0, node.getHeight() - reference2.getHeight());
        Node movingAttachmentParent = node.getParent();

        // set up optimal transport

        Set<Double> pathHeights = this.getPathHeights(reference1, reference2, node, movingAttachmentParent);

        UnivariateOptimalTransportMap transportMap = new UnivariateOptimalTransportMap(
                minAttachmentDistance, maxAttachmentDistance, pathHeights, height -> {
                    TreeUtils.reattachNode(node, height, reference1, reference2);
                    double logDensity = computeCurrentLogLikelihood.get();

                    // the cache now matches this tree, so the next evaluation only recalculates what the
                    // next reattachment changes. Only the node flags are cleared: the tree stays marked as
                    // edited, otherwise its next change would overwrite the backup a rejection restores
                    this.tree.getRoot().makeAllDirty(Tree.IS_CLEAN);

                    return logDensity;
                }
        );

        double[] proposal = this.getProposal(transportMap, currentDistance, pathHeights);

        if (proposal == null) {
            // the cache now matches the tree of the last grid evaluation
            this.affectedNodeNrs = this.getNodesAffectedByReattachment(node, originalSibling);
            return Double.NEGATIVE_INFINITY;
        }

        if (debug) {
            int max = 100;
            for (int i = 0; i < max - 1; i++) {
                double h = minAttachmentDistance + (i + 1.0) / max * (maxAttachmentDistance - minAttachmentDistance);
                TreeUtils.reattachNode(node, h, reference1, reference2);

                System.out.println("approx," + h + "," + computeCurrentLogLikelihood.get() + "," + transportMap.logDensity(h));
            }

            for (int i = 0; i < 1000; i++) {
                double sampledDistance = transportMap.sample();
                TreeUtils.reattachNode(node, sampledDistance, reference1, reference2);

                System.out.println("sample," + sampledDistance + "," + computeCurrentLogLikelihood.get() + "," + transportMap.logDensity(sampledDistance));
            }

            System.exit(1);
        }

        // move to the sampled attachment

        double sampledDistance = proposal[0];
        TreeUtils.reattachNode(node, sampledDistance, reference1, reference2);

        // the proposed tree is the one the cache will hold once the MCMC has evaluated it
        this.affectedNodeNrs = this.getNodesAffectedByReattachment(node, originalSibling);

        // compute log HR ratio

        double backwardHR = this.getTripletLogProbability(reference1, reference2, node);

        return proposal[1] + backwardHR - forwardHR;
    }

    /**
     * Draws the new attachment distance and returns it together with the log Hastings ratio of the
     * map, or null when the current attachment cannot be moved.
     *
     * <p>By default only attachments on another branch are proposed: the map usually puts most of its
     * mass on the branch the node already hangs on, and moving it within that branch is left to the
     * cheaper height operators. With {@code sampleCurrentBranch} the whole path is sampled instead.
     */
    private double[] getProposal(UnivariateOptimalTransportMap transportMap, double currentDistance, Set<Double> pathHeights) {
        if (!transportMap.isInsideSamplingSupport(currentDistance)) {
            return null;
        }

        if (this.sampleCurrentBranch) {
            double sampledDistance = transportMap.sample();
            return new double[] {sampledDistance, transportMap.logHRCorrection(currentDistance, sampledDistance)};
        }

        double[] branchBoundaries = pathHeights.stream()
                .filter(transportMap::isInsideSamplingSupport)
                .mapToDouble(Double::doubleValue)
                .sorted()
                .toArray();

        return transportMap.sampleExcludingSegment(currentDistance, branchBoundaries);
    }

    /**
     * Marks only the nodes the rejected proposal can have left stale. The cache holds the tree the
     * node was last evaluated in, which differs from the restored tree only in where the node hangs.
     * The partials that differ are therefore those of the ancestors of the node's parent in the
     * restored tree and of the ancestors of the branch it hung on in the cached tree. Marking the
     * node, its parent and the two siblings is enough, since the tree likelihood recomputes the
     * partials of every ancestor of a dirty node, which is the least any recalculation can do.
     */
    @Override
    public void markDirtyAfterRestore(State state) {
        if (this.affectedNodeNrs == null) {
            super.markDirtyAfterRestore(state);
            return;
        }

        for (int nodeNr : this.affectedNodeNrs) {
            this.tree.getNode(nodeNr).makeDirty(Tree.IS_FILTHY);
        }

        // restore() cleared the tree's edited flag, without it no likelihood would recalculate
        this.tree.setSomethingIsDirty(true);
    }

    /**
     * Returns the numbers of the moved node, its parent, its sibling before the proposal and its
     * sibling in the cached tree. Node numbers survive the restore, which swaps the node objects.
     */
    private int[] getNodesAffectedByReattachment(Node node, Node originalSibling) {
        Node parent = node.getParent();
        Node cachedSibling = TreeUtils.getOtherChild(parent, node);

        return new int[] {node.getNr(), parent.getNr(), originalSibling.getNr(), cachedSibling.getNr()};
    }

    private record WeightedCandidates(List<Node> candidates, List<Double> weights) {}

    private WeightedCandidates getWeightedNodeCandidates(Node reference1, Node reference2) {
        List<Node> candidates = this.getNodeToMoveCandidates(reference1, reference2);
        List<Double> weights = new ArrayList<>(candidates.size());

        for (Node candidate : candidates) {
            weights.add(this.getWeight(reference1, reference2, candidate));
        }

        return new WeightedCandidates(candidates, weights);
    }

    // recomputes the candidates and weights from scratch. Used for the backward direction, where
    // the tree has already been mutated by the move and the candidates may genuinely have changed

    private double getTripletLogProbability(Node reference1, Node reference2, Node node) {
        List<Node> reference2Candidates = this.getReference2Candidates(reference1);
        WeightedCandidates nodeCandidates = this.getWeightedNodeCandidates(reference1, reference2);

        return this.getTripletLogProbability(reference2Candidates, reference2, nodeCandidates, node);
    }

    // uses already computed candidates and weights. Used for the forward direction, where these
    // are identical to what a fresh computation would produce since the tree has not moved yet

    private double getTripletLogProbability(
            List<Node> reference2Candidates, Node reference2, WeightedCandidates nodeCandidates, Node node) {
        // compute probability for reference 2

        if (!reference2Candidates.contains(reference2)) {
            throw new IllegalArgumentException("reference2 not amongst the candidates for it.");
        }

        double logReference2Probability = Math.log(1.0 / reference2Candidates.size());

        // compute probability for node

        List<Node> candidates = nodeCandidates.candidates();
        List<Double> weights = nodeCandidates.weights();

        double cumWeight = 0.0;
        double nodeWeight = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < candidates.size(); i++) {
            double weight = weights.get(i);
            cumWeight += weight;

            if (candidates.get(i) == node) {
                nodeWeight = weight;
            }
        }

        if (nodeWeight == Double.NEGATIVE_INFINITY) {
            throw new IllegalArgumentException("Node not amongst the candidates for it.");
        }

        double logNodeProbability = Math.log(nodeWeight / cumWeight);

        return logReference2Probability + logNodeProbability;
    }

    private List<Node> getReference2Candidates(Node reference1) {
        List<Node> candidates = new ArrayList<>();

        for (int nodeId = 0; nodeId < this.tree.getLeafNodeCount(); nodeId++) {
            Node candidate = this.tree.getNode(nodeId);

            if (reference1.getParent() == candidate.getParent()) {
                continue;
            }

            candidates.add(candidate);
        }

        return candidates;
    }

    private List<Node> getNodeToMoveCandidates(Node reference1, Node reference2) {
        Set<Node> path = TreeUtils.getCommonAncestor(reference1, reference2).path();

        List<Node> candidates = new ArrayList<>();
        for (Node parent : path) {
            if (parent.isLeaf()) continue;

            Node left = parent.getLeft();
            if (!TreeUtils.containsNode(left, reference1) && !TreeUtils.containsNode(left, reference2)) {
                candidates.add(left);
            }

            Node right = parent.getRight();
            if (!TreeUtils.containsNode(right, reference1) && !TreeUtils.containsNode(right, reference2)) {
                candidates.add(right);
            }
        }

        return candidates;
    }

    private double getWeight(Node reference1, Node reference2, Node nodeToMove) {
        double distance1 = this.getMeanTaxaDistance(reference1, nodeToMove);
        double distance2 = this.getMeanTaxaDistance(reference2, nodeToMove);

        double sumDistance = distance1 + distance2;
        double referenceDistance = this.distance.pairwiseDistance(reference1.getNr(), reference2.getNr());
        double tension = referenceDistance / Math.max(MIN_DISTANCE, sumDistance);
        double angleDenominator = Math.max(MIN_DISTANCE, 2.0 * distance1 * distance2);
        double cosAngle = (distance1 * distance1 + distance2 * distance2 - referenceDistance * referenceDistance)
                / angleDenominator;
        double angle = Math.acos(Math.max(-1.0, Math.min(1.0, cosAngle)));
        double sequencePosition = distance1 / Math.max(MIN_DISTANCE, sumDistance);
        double treePosition = TreeUtils.getDistance(reference1, nodeToMove.getParent())
                / Math.max(MIN_DISTANCE, TreeUtils.getDistance(reference1, reference2));
        double displacement = Math.abs(sequencePosition - treePosition);

        return switch (this.weightingScheme) {
            case UNIFORM -> 1.0;
            case PROP -> sumDistance;
            case DISPLACEMENT -> MIN_DISTANCE + displacement;
            case TENSION -> MIN_DISTANCE + tension;
            case ANGLE -> MIN_DISTANCE + angle;
            case INVERSE -> 1.0 / sumDistance;
            case INVERSE_SQUARE -> 1.0 / (sumDistance * sumDistance);
            case INVERSE_SQRT -> 1.0 / Math.sqrt(sumDistance);
        };
    }

    private double getMeanTaxaDistance(Node reference, Node subtree) {
        double distanceSum = 0.0;
        int count = 0;

        for (Node leaf : subtree.getAllLeafNodes()) {
            distanceSum += this.distance.pairwiseDistance(reference.getNr(), leaf.getNr());
            count++;
        }

        if (subtree.isLeaf()) {
            count = 1;
            distanceSum = this.distance.pairwiseDistance(reference.getNr(), subtree.getNr());
        }

        return distanceSum / count;
    }

    private int sampleWeightedIndex(List<Double> weights) {
        double totalWeight = 0.0;
        for (double weight : weights) {
            totalWeight += weight;
        }

        double threshold = Randomizer.nextDouble() * totalWeight;
        double cumulativeWeight = 0.0;
        for (int i = 0; i < weights.size(); i++) {
            cumulativeWeight += weights.get(i);
            if (threshold < cumulativeWeight) {
                return i;
            }
        }

        return weights.size() - 1;
    }

    private enum WeightingScheme {
        UNIFORM,
        PROP,
        DISPLACEMENT,
        TENSION,
        ANGLE,
        INVERSE,
        INVERSE_SQRT,
        INVERSE_SQUARE;

        private static WeightingScheme fromName(String name) {
            try {
                return WeightingScheme.valueOf(name.toUpperCase());
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException("Unknown weightingScheme '" + name, exception);
            }
        }
    }

    private Set<Double> getPathHeights(Node reference1, Node reference2, Node node, Node movingAttachmentParent) {
        TreeUtils.MRCA mrca = TreeUtils.getCommonAncestor(reference1, reference2);
        Set<Node> path = mrca.path();
        path.remove(reference1);
        path.remove(reference2);
        path.remove(movingAttachmentParent);
        path.add(mrca.mrca());
        return path.stream()
                .filter(x -> node.getHeight() < x.getHeight())
                .map(x -> TreeUtils.getDistance(reference1, x))
                .collect(Collectors.toSet());
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
