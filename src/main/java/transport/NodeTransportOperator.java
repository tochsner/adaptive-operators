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

import java.util.*;
import java.util.function.Supplier;

public class NodeTransportOperator extends SliceOperator {

    public Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");

    int BURN_IN = 0;

    Tree tree;
    Alignment alignment;
    private boolean debug = false;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();
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

        // select a reference pair and a subtree that does not contain either reference

        Node reference1 = null;
        Node reference2 = null;
        Node node = null;
        int maxAttempts = 100;

        for (int attempt = 0; attempt < maxAttempts && reference2 == null; attempt++) {
            int referenceId1 = Randomizer.nextInt(this.tree.getLeafNodeCount());
            reference1 = this.tree.getNode(referenceId1);
            node = reference1.getParent().getLeft() == reference1
                    ? reference1.getParent().getRight()
                    : reference1.getParent().getLeft();

            int referenceId2 = Randomizer.nextInt(this.tree.getLeafNodeCount());
            Node candidateReference2 = this.tree.getNode(referenceId2);
            if (candidateReference2 != reference1 && !TreeUtils.containsNode(node, candidateReference2)) {
                reference2 = candidateReference2;
            }
        }

        if (reference2 == null) {
            return Double.NEGATIVE_INFINITY;
        }

        final Node selectedReference1 = reference1;
        final Node selectedReference2 = reference2;
        final Node selectedNode = node;

        // get the current and max distance

        double currentDistance = selectedNode.getParent().getHeight() - selectedReference1.getHeight();
        double maxDistance = TreeUtils.getDistance(selectedReference1, selectedReference2);
        double minAttachmentDistance = Math.max(0.0, selectedNode.getHeight() - selectedReference1.getHeight());
        double maxAttachmentDistance = maxDistance - Math.max(0.0, selectedNode.getHeight() - selectedReference2.getHeight());

        if (maxAttachmentDistance <= minAttachmentDistance
                || currentDistance <= minAttachmentDistance
                || currentDistance >= maxAttachmentDistance) {
            return Double.NEGATIVE_INFINITY;
        }

        // set up optimal transport

        UnivariateOptimalTransportMap transportMap = new UnivariateOptimalTransportMap(
                minAttachmentDistance, maxAttachmentDistance, height -> {
                    TreeUtils.reattachNode(selectedNode, height, selectedReference1, selectedReference2);
                    return computeCurrentLogLikelihood.get();
                }
        );

        if (debug) {
            int max = 100;
            for (int i = 0; i < max - 1; i++) {
                double h = minAttachmentDistance + (i + 1.0) / max * (maxAttachmentDistance - minAttachmentDistance);
                TreeUtils.reattachNode(selectedNode, h, selectedReference1, selectedReference2);

                System.out.println("approx," + h + "," + computeCurrentLogLikelihood.get() + "," + transportMap.logDensity(h));
            }

            for (int i = 0; i < 1000; i++) {
                double sampledDistance = transportMap.sample();
                TreeUtils.reattachNode(selectedNode, sampledDistance, selectedReference1, selectedReference2);

                System.out.println("sample," + sampledDistance + "," + computeCurrentLogLikelihood.get() + "," + transportMap.logDensity(sampledDistance));
            }

            System.exit(1);
        }

        // sample

        double sampledDistance = transportMap.sample();
        TreeUtils.reattachNode(selectedNode, sampledDistance, selectedReference1, selectedReference2);

        return transportMap.logHRCorrection(currentDistance, sampledDistance);
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
