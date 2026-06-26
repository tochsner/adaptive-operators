package transport;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.distance.JukesCantorDistance;
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
    public final Input<String> weightingSchemeInput = new Input<>(
            "weightingScheme",
            "taxon weighting scheme: inverse, inverse_sqrt, or explorative_inverse_sqrt",
            "inverse"
    );

    int BURN_IN = 0;

    Tree tree;
    Alignment alignment;
    private boolean debug = false;
    JukesCantorDistance distance;
    WeightingScheme weightingScheme;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();

        this.distance = new JukesCantorDistance();
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

        // select the second reference in a weighted manner and such that it is not the direct sibling

        Node reference2 = this.sampleReference2(reference1);

        // select the node to move on the path between the two references

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
        List<Node> candidates = new ArrayList<>();
        List<Double> weights = new ArrayList<>();

        for (int nodeId = 0; nodeId < this.tree.getLeafNodeCount(); nodeId++) {
            Node candidate = this.tree.getNode(nodeId);

            if (reference1.getParent() == candidate.getParent()) {
                continue;
            }

            double distanceToReference = this.distance.pairwiseDistance(reference1.getNr(), nodeId);
            candidates.add(candidate);
            weights.add(this.getWeight(reference1, candidate, distanceToReference));
        }

        int index = this.sampleWeightedIndex(weights);
        return candidates.get(index);
    }

    private Node sampleNodeToMove(Node reference1, Node reference2) {
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

        return candidates.get(Randomizer.nextInt(candidates.size()));
    }

    private double getTripletLogProbability(Node reference1, Node reference2, Node node) {
        // compute probability for reference 2

        double reference2Weight = 0.0;
        double cumWeight = 0.0;

        for (int nodeId = 0; nodeId < this.tree.getLeafNodeCount(); nodeId++) {
            Node candidate = this.tree.getNode(nodeId);

            if (reference1.getParent() == candidate.getParent()) {
                continue;
            }

            double distanceToReference = this.distance.pairwiseDistance(reference1.getNr(), nodeId);
            double weight = this.getWeight(reference1, candidate, distanceToReference);

            if (candidate == reference2) reference2Weight = weight;
            cumWeight += weight;

        }

        if (reference2Weight == 0.0) {
            throw new IllegalArgumentException("Reference 2 not amongst the candidates for it.");
        }

        double logReference2Probability = Math.log(reference2Weight / cumWeight);

        // compute probability for node

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

        if (!candidates.contains(node)) {
            throw new IllegalArgumentException("Node not amongst the candidates for it.");
        }

        double logNodeProbability = Math.log(1.0 / candidates.size());

        return logReference2Probability + logNodeProbability;
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

    private double getWeight(Node reference, Node candidate, double sequenceDistance) {
        return switch (this.weightingScheme) {
            case UNIFORM -> 1.0;
            case INVERSE -> 1.0 / sequenceDistance;
            case INVERSE_SQUARE -> 1.0 / (sequenceDistance * sequenceDistance);
            case INVERSE_SQRT -> 1.0 / Math.sqrt(sequenceDistance);
            case EXPLORATIVE_INVERSE_SQRT -> {
                double treeDistance = Math.max(TreeUtils.getDistance(reference, candidate), 1e-12);
                yield Math.pow(treeDistance, 0.2) / sequenceDistance;
            }
        };
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
        INVERSE,
        INVERSE_SQRT,
        INVERSE_SQUARE,
        EXPLORATIVE_INVERSE_SQRT;

        private static WeightingScheme fromName(String name) {
            try {
                return WeightingScheme.valueOf(name.toUpperCase());
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException("Unknown weightingScheme '" + name, exception);
            }
        }
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
