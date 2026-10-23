package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import surrogate.RobinsonFouldsTreeMetric;

import java.util.List;

/**
 * Immutable adapter that summarises a tree by its scaled Robinson-Foulds distance to a reference
 * tree.
 *
 * <p>The result is a single real scalar and so serves as a cheap proxy trace for the mixing of the
 * tree topology. The reference tree is captured lazily, on the first read, so that it is the tree
 * the chain actually starts from rather than whatever the tree held while the analysis was still
 * being initialised.
 *
 * <p>A starting tree far from the posterior shares almost no clades with the trees the chain
 * visits later, so the distance to it saturates and stops reflecting how the topology moves. The
 * reference should therefore be moved to a tree from the posterior through {@link #resetReference()}
 * once the chain has settled.
 */
public class RobinsonFouldsDistanceAdapter extends BEASTObject implements ReferenceAdapter {

    public final Input<Tree> treeInput = new Input<>("tree", "Tree to measure the distance of.");

    private Tree tree;
    private RobinsonFouldsTreeMetric metric;
    private long[] referenceEmbedding;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.metric = new RobinsonFouldsTreeMetric(this.tree.getLeafNodeCount());
    }

    @Override
    public int getNumImmutable() {
        return 1;
    }

    @Override
    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        long[] embedding = this.metric.embed(this.tree);

        if (this.referenceEmbedding == null) {
            this.referenceEmbedding = embedding;
        }

        return new double[]{RobinsonFouldsTreeMetric.distance(this.referenceEmbedding, embedding)};
    }

    @Override
    public void resetReference() {
        this.referenceEmbedding = this.metric.embed(this.tree);
    }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double update(double[] mutable, int nodeId) {
        return 0.0;
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0.0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }
}
