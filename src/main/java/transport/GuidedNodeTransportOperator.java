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
    private static final double MIN_DISTANCE = 1.0E-12;

    int BURN_IN = 0;

    Tree tree;
    Alignment alignment;
    private boolean debug = false;
    FastJCDistance distance;
    WeightingScheme weightingScheme;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();

        this.distance = new FastJCDistance();
        this.distance.setPatterns(this.alignment);
        this.weightingScheme = WeightingScheme.fromName(this.weightingSchemeInput.get());
    }

    int count = 0;

    @Override
    public double proposal() {
        throw new UnsupportedOperationException();
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood, State state) {
        this.count++;
        if (this.count < BURN_IN) return Double.NEGATIVE_INFINITY;
        else if (this.count == BURN_IN) System.out.println("Start transport");

        // select the first reference at random

        int referenceId1 = Randomizer.nextInt(this.tree.getLeafNodeCount());
        Node reference1 = this.tree.getNode(referenceId1);

        // select the second reference such that it is not the direct sibling

        Node reference2 = this.sampleReference2(reference1);

        // select the node to move on the path between the two references in a weighted manner

        Node node = this.sampleNodeToMove(reference1, reference2);

        // compute the old triplet selection density

        double forwardHR = this.getTripletLogProbability(reference1, reference2, node);

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
                    Tree t = tree.copy();
                    return computeCurrentLogLikelihood.get();
                }
        );

        if (!transportMap.isInsideSamplingSupport(currentDistance)) {
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

        // sample

        double sampledDistance = transportMap.sample();
        TreeUtils.reattachNode(node, sampledDistance, reference1, reference2);

        // compute log HR ratio

        double backwardHR = this.getTripletLogProbability(reference1, reference2, node);

        double logHR = transportMap.logHRCorrection(currentDistance, sampledDistance);
        logHR += backwardHR - forwardHR;

        return logHR;
    }

    private Node sampleReference2(Node reference1) {
        List<Node> candidates = this.getReference2Candidates(reference1);
        return candidates.get(Randomizer.nextInt(candidates.size()));
    }

    private Node sampleNodeToMove(Node reference1, Node reference2) {
        List<Node> candidates = this.getNodeToMoveCandidates(reference1, reference2);
        List<Double> weights = new ArrayList<>();

        for (Node candidate : candidates) {
            double weight = this.getWeight(reference1, reference2, candidate);
            weights.add(weight);
        }

        int candidateIdx = this.sampleWeightedIndex(weights);
        return candidates.get(candidateIdx);
    }

    private double getTripletLogProbability(Node reference1, Node reference2, Node node) {
        // compute probability for reference 2

        List<Node> reference2Candidates = this.getReference2Candidates(reference1);
        if (!reference2Candidates.contains(reference2)) {
            throw new IllegalArgumentException("reference2 not amongst the candidates for it.");
        }

        double logReference2Probability = Math.log(1.0 / reference2Candidates.size());

        // compute probability for node

        List<Node> nodeCandidates = this.getNodeToMoveCandidates(reference1, reference2);

        double cumWeight = 0.0;
        double nodeWeight = Double.NEGATIVE_INFINITY;

        for (Node nodeCandidate : nodeCandidates) {
            double weight = this.getWeight(reference1, reference2, nodeCandidate);
            cumWeight += weight;

            if (nodeCandidate == node) {
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
