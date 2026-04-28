package adaptiveoperators;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

public class MyScaleOperator extends Operator {

    final public Input<Tree> treeInput = new Input<>(
            "tree",
            "the tree to operate one",
            Input.Validate.REQUIRED);

    @Override
    public void initAndValidate() {
    }

    @Override
    public double proposal() {
        Tree tree = this.treeInput.get();

        Node node = tree.getNode(Randomizer.nextInt(tree.getNodeCount()));
        while (node.isRoot()) {
            node = tree.getNode(Randomizer.nextInt(tree.getNodeCount()));
        }

        Node parent = node.getParent();

        // we sample the branch length from conditional

        return 0;
    }

}
