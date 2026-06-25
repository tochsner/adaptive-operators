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
import org.apache.commons.math4.legacy.core.Pair;
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

        // select three nodes in a guided manner

        int node1Id = Randomizer.nextInt(this.tree.getLeafNodeCount());
        Node node1 = this.tree.getNode(node1Id);

        Pair<Node, Node> similar = this.sampleSimilarNodes(node1);
        Node node2 = similar.getFirst();
        Node node3 = similar.getSecond();

        // the three form a triplet (A,BC)
        Triplet triplet = this.orientTriplet(node1, node2, node3);
        Node reference1 = triplet.outgroup();
        Node reference2;
        Node node;

        if (Randomizer.nextBoolean()) {
            reference2 = triplet.firstCherryNode();
            node = triplet.secondCherryNode();
        } else {
            reference2 = triplet.secondCherryNode();
            node = triplet.firstCherryNode();
        }

        // get the current and max distance

        double currentDistance = TreeUtils.getDistance(reference1, node.getParent());
        double maxDistance = TreeUtils.getDistance(reference1, reference2);
        double minAttachmentDistance = Math.max(0.0, node.getHeight() - reference1.getHeight());
        double maxAttachmentDistance = maxDistance - Math.max(0.0, node.getHeight() - reference2.getHeight());

        if (maxAttachmentDistance <= minAttachmentDistance
                || currentDistance <= minAttachmentDistance
                || currentDistance >= maxAttachmentDistance) {
            return Double.NEGATIVE_INFINITY;
        }

        // set up optimal transport

        Set<Double> pathHeights = this.getPathHeights(reference1, reference2, node);

        UnivariateOptimalTransportMap transportMap = new UnivariateOptimalTransportMap(
                minAttachmentDistance, maxAttachmentDistance, pathHeights, height -> {
                    TreeUtils.reattachNode(node, height, reference1, reference2);
                    return computeCurrentLogLikelihood.get();
                }
        );

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

        return transportMap.logHRCorrection(currentDistance, sampledDistance);
    }

    private Set<Double> getPathHeights(Node reference1, Node reference2, Node node) {
        TreeUtils.MRCA mrca = TreeUtils.getCommonAncestor(reference1, reference2);
        Set<Node> path = mrca.path();
        path.remove(reference1);
        path.remove(reference2);
        path.remove(TreeUtils.getCommonAncestor(reference1, node).mrca());
        path.add(mrca.mrca());
        return path.stream()
                .filter(x -> node.getHeight() < x.getHeight())
                .map(x -> TreeUtils.getDistance(reference1, x))
                .collect(Collectors.toSet());
    }

    private Pair<Node, Node> sampleSimilarNodes(Node reference) {
        // compute distances of all nodes to reference
        // sample two other nodes with no replacement weighted by the inverse distance (more similar more often)
        List<Node> candidates = new ArrayList<>();
        List<Double> weights = new ArrayList<>();

        for (int nodeId = 0; nodeId < this.tree.getLeafNodeCount(); nodeId++) {
            if (nodeId == reference.getNr()) {
                continue;
            }

            double distanceToReference = this.distance.pairwiseDistance(reference.getNr(), nodeId);
            if (!Double.isFinite(distanceToReference)) {
                continue;
            }

            Node candidate = this.tree.getNode(nodeId);
            candidates.add(candidate);
            weights.add(this.getWeight(reference, candidate, distanceToReference));
        }

        if (candidates.size() < 2) {
            throw new IllegalArgumentException("GuidedNodeTransportOperator requires at least three finite-distance taxa");
        }

        int firstIndex = this.sampleWeightedIndex(weights);
        Node first = candidates.remove(firstIndex);
        weights.remove(firstIndex);

        int secondIndex = this.sampleWeightedIndex(weights);
        Node second = candidates.get(secondIndex);

        return Pair.create(first, second);
    }

    private double getWeight(Node reference, Node candidate, double sequenceDistance) {
        double safeSequenceDistance = Math.max(sequenceDistance, 1e-12);

        return switch (this.weightingScheme) {
            case INVERSE -> 1.0 / safeSequenceDistance;
            case INVERSE_SQRT -> 1.0 / Math.sqrt(safeSequenceDistance);
            case EXPLORATIVE_INVERSE_SQRT -> {
                double treeDistance = Math.max(TreeUtils.getDistance(reference, candidate), 1e-12);
                yield Math.pow(treeDistance, 0.2) / safeSequenceDistance;
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

    private Triplet orientTriplet(Node node1, Node node2, Node node3) {
        double height12 = TreeUtils.getCommonAncestor(node1, node2).mrca().getHeight();
        double height13 = TreeUtils.getCommonAncestor(node1, node3).mrca().getHeight();
        double height23 = TreeUtils.getCommonAncestor(node2, node3).mrca().getHeight();

        if (height12 < height13 && height12 < height23) {
            return new Triplet(node3, node1, node2);
        }

        if (height13 < height12 && height13 < height23) {
            return new Triplet(node2, node1, node3);
        }

        if (height23 < height12 && height23 < height13) {
            return new Triplet(node1, node2, node3);
        }

        throw new IllegalArgumentException("Could not orient triplet with a unique cherry");
    }

    private record Triplet(Node outgroup, Node firstCherryNode, Node secondCherryNode) {}

    private enum WeightingScheme {
        INVERSE,
        INVERSE_SQRT,
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
