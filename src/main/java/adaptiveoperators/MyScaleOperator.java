package adaptiveoperators;

import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

import java.util.List;

public class MyScaleOperator extends Operator {

    final public Input<Tree> treeInput = new Input<>(
            "tree",
            "the tree to operate one",
            Input.Validate.REQUIRED);

    final public Input<List<Function>> conditionsInput = new Input<>(
            "conditions",
            "the conditions to operate one",
            Input.Validate.REQUIRED);

    ConditionalAdaptiveSampler sampler;

    @Override
    public void initAndValidate() {
        List<Function> conditions = this.conditionsInput.get();

        this.sampler = new MultivariateNormalSampler(conditions.size(), 1);
    }

    @Override
    public double proposal() {
        Tree tree = this.treeInput.get();
        List<Function> conditions = this.conditionsInput.get();

        // record values for all nodes

        double[] conditionValues = new double[conditions.size()];
        for (int i = 0; i < conditionValues.length; i++) {
            conditionValues[i] = conditions.get(i).getArrayValue();
        }

        for (Node candidateNode : tree.getNodesAsArray()) {
            if (candidateNode.isRoot()) continue;

            double[] values = new double[] {candidateNode.getParent().getHeight() - candidateNode.getHeight()};
            sampler.record(conditionValues, values);
        }

        // update random node

        Node node = tree.getNode(Randomizer.nextInt(tree.getNodeCount()));
        while (node.isRoot()) {
            node = tree.getNode(Randomizer.nextInt(tree.getNodeCount()));
        }

        Node parent = node.getParent();
        double oldBranchLength = parent.getHeight() - node.getHeight();

        // we sample the branch length from conditional

        double branchLength = this.sampler.sampleConditionally(conditionValues)[0];
        if (branchLength < 0) return Double.NEGATIVE_INFINITY;

        // set branch length

        parent.setHeight(
                node.getHeight() + branchLength
        );

        // TODO what is the HR?

        return 0;
    }

}
