package surrogate;

import adapters.Adapter;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Tree;
import beast.base.inference.*;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.evolution.tree.ClusterTree;
import beast.base.util.Randomizer;
import mala.SubsetJukesCantorDistance;

import java.util.ArrayList;
import java.util.List;

public class TreeGraphOperator extends TreeOperator {

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    public final Input<Integer> ballRadiusInput = new Input<>("ballRadius",
            "how many hops of the graph a single jump may cover", 1);

    Tree tree;
    List<Adapter> adapters;
    Alignment alignment;
    int ballRadius;

    List<SampledTree> topologySamples;
    TreeGraph treeGraph;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.adapters = this.adaptersInput.get();
        this.alignment = this.alignmentInput.get();
        this.ballRadius = this.ballRadiusInput.get();

        this.topologySamples = this.sampleTopologies();
        this.treeGraph = new TreeGraph(this.topologySamples);
    }

    private List<SampledTree> sampleTopologies() {
        int numTopologies = 1_000;
        double sitesFraction = 0.75;

        JukesCantor substitutionModel = new JukesCantor();
        substitutionModel.initAndValidate();

        SiteModel siteModel = new SiteModel();
        siteModel.substModelInput.setTypedValue(substitutionModel, siteModel);
        siteModel.initAndValidate();

        TreeLikelihood treeLikelihood = new TreeLikelihood();
        treeLikelihood.dataInput.setTypedValue(this.alignment, treeLikelihood);
        treeLikelihood.siteModelInput.setTypedValue(siteModel, treeLikelihood);

        List<SampledTree> sampledTopologies = new ArrayList<>();
        for (int i = 0; i < numTopologies; i++) {
            ClusterTree tree = new ClusterTree();
            tree.clusterTypeInput.setTypedValue(ClusterTree.Type.upgma, tree);
            tree.distanceInput.setTypedValue(new SubsetJukesCantorDistance(sitesFraction), tree);
            tree.dataInput.setTypedValue(this.alignment, tree);
            tree.m_traitList.setTypedValue(this.tree.m_traitList.get(), tree);
            tree.initAndValidate();

            treeLikelihood.treeInput.setTypedValue(tree, treeLikelihood);
            treeLikelihood.initAndValidate();

            double logLikelihood = treeLikelihood.calculateLogP();
            sampledTopologies.add(new SampledTree(tree, logLikelihood));
        }

        return sampledTopologies;
    }

    @Override
    public double proposal() {
        // find the closest reference tree

        SampledTree closestTree = this.treeGraph.getClosest(this.tree);

        double oldHeight = this.tree.getRoot().getHeight();

        this.tree.startEditing(this);
        this.tree.assignFromFragile(closestTree.topology());

        double newHeight = this.tree.getRoot().getHeight();
        this.tree.scale(oldHeight / newHeight);

        return 0;

        // randomly pick a node from the ball around it

        // List<SampledTree> sourceBall = this.treeGraph.getBall(closestTree, this.ballRadius);
        // SampledTree jumpTarget = sourceBall.get(Randomizer.nextInt(sourceBall.size()));


        // the relaxation slides along the offset of the state tree from its own projection, which
        // is scale free because both embeddings are divided by their own root height

//        double[] direction = this.getRelaxationDirection(closestTree);
//
//        // perturb new tree
//
//        double rootHeight = this.tree.getRoot().getHeight();
//        double[] anchor = this.treeGraph.getHeightEmbedding(jumpTarget);
//        double[] forwardBounds = RelativeHeightEmbedding.segmentBounds(
//                anchor, direction, RelativeHeightEmbedding.minimumRelativeHeights(jumpTarget.topology(), rootHeight)
//        );
//
//        double step = forwardBounds[0] + Randomizer.nextDouble() * (forwardBounds[1] - forwardBounds[0]);
//        step = 0;
//
//        double[] proposed = new double[anchor.length];
//        for (int i = 0; i < proposed.length; i++) {
//            proposed[i] = anchor[i] + step * direction[i];
//        }
//
//        this.tree.startEditing(this);
//        RelativeHeightEmbedding.decode(this.tree, jumpTarget.topology(), proposed, rootHeight);
//
//        // a jump only counts if the perturbed tree still projects onto the node it was drawn
//        // around
//        if (!this.treeGraph.isClosest(this.tree, jumpTarget)) {
//            return Double.NEGATIVE_INFINITY;
//        }
//
//        // compute log hastings ratio
//
//        // the reverse move runs the same line through the old projection. Its direction is the
//        // negated one, which leaves the length of the admissible interval unchanged.
//        //
//        // this assumes the state tree carries the topology of its own projection, which holds for
//        // every tree this operator itself produced, since the check above only lets through trees
//        // that project onto the node whose topology they were given. A state tree that arrived
//        // from another operator with a different topology is strictly unreachable by the reverse
//        // move, and the ratio below overstates it.
//        double[] reverseBounds = RelativeHeightEmbedding.segmentBounds(
//                this.treeGraph.getHeightEmbedding(closestTree), direction,
//                RelativeHeightEmbedding.minimumRelativeHeights(closestTree.topology(), rootHeight)
//        );
//
//        List<SampledTree> targetBall = this.treeGraph.getBall(jumpTarget, this.ballRadius);
//
//        return 0;

        // return Math.log(targetBall.size()) - Math.log(targetBall.size())
        //         + Math.log(forwardBounds[1] - forwardBounds[0])
         //        - Math.log(reverseBounds[1] - reverseBounds[0]);
    }

    /**
     * Returns the unit vector pointing from the projection of the state tree to the state tree
     * itself.
     */
    private double[] getRelaxationDirection(SampledTree closestTree) {
        double[] current = RelativeHeightEmbedding.encode(this.tree);
        double[] projection = this.treeGraph.getHeightEmbedding(closestTree);

        double[] direction = new double[current.length];
        double sumOfSquares = 0.0;

        for (int i = 0; i < direction.length; i++) {
            direction[i] = current[i] - projection[i];
            sumOfSquares += direction[i] * direction[i];
        }

        double norm = Math.sqrt(sumOfSquares);
        for (int i = 0; i < direction.length; i++) {
            direction[i] /= norm;
        }

        return direction;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
