package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.List;

/**
 * Per-node tree embedding adapter.
 *
 * <p>Computes a fixed-length embedding vector for each node in a tree by propagating random leaf
 * embeddings up the tree via a fixed global random linear transformation. Each internal node's
 * embedding is computed by averaging its children's embeddings, multiplying by the fixed
 * projection matrix, and normalizing to unit length.
 *
 * <p>The embedding process is deterministic given the tree topology and the fixed random
 * state (leaf embeddings and projection matrix), both generated once at initialization using
 * BEAST's global seeded Randomizer.
 */
public class NodeEmbeddingAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Integer> dimensionInput = new Input<>("dimension",
            "dimension of the per-node embedding", 16);

    private Tree tree;
    private int dimension;
    private double[][] leafEmbeddings;
    private double[][] projectionMatrix;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.dimension = this.dimensionInput.get();

        if (this.dimension <= 0) {
            throw new IllegalArgumentException("embedding dimension must be positive");
        }

        int leafCount = this.tree.getLeafNodeCount();

        // generate random leaf embeddings (one per taxon), each normalized to unit length
        this.leafEmbeddings = new double[leafCount][this.dimension];
        for (int i = 0; i < leafCount; i++) {
            for (int d = 0; d < this.dimension; d++) {
                this.leafEmbeddings[i][d] = Randomizer.nextGaussian();
            }
            normalize(this.leafEmbeddings[i]);
        }

        // generate random projection matrix (fixed global dxd matrix)
        this.projectionMatrix = new double[this.dimension][this.dimension];
        for (int i = 0; i < this.dimension; i++) {
            for (int j = 0; j < this.dimension; j++) {
                this.projectionMatrix[i][j] = Randomizer.nextGaussian();
            }
        }
    }

    @Override
    public int getNumImmutable() {
        return this.dimension;
    }

    @Override
    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        Node node = this.tree.getNode(nodeId);
        return this.computeEmbedding(node);
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

    /**
     * Recursively compute the embedding for a node.
     */
    private double[] computeEmbedding(Node node) {
        if (node.isLeaf()) {
            // return a copy of the pre-generated leaf embedding
            return this.leafEmbeddings[node.getNr()].clone();
        }

        // internal node: average children's embeddings
        List<Node> children = node.getChildren();
        double[] averaged = new double[this.dimension];

        for (Node child : children) {
            double[] childEmbedding = this.computeEmbedding(child);
            for (int d = 0; d < this.dimension; d++) {
                averaged[d] += childEmbedding[d];
            }
        }

        // divide by number of children to complete the average
        for (int d = 0; d < this.dimension; d++) {
            averaged[d] /= children.size();
        }

        // multiply by projection matrix
        double[] projected = this.multiplyMatrixVector(this.projectionMatrix, averaged);

        // normalize to unit length
        normalize(projected);

        return projected;
    }

    /**
     * Multiply a d x d matrix by a d-dimensional vector.
     */
    private double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int d = vector.length;
        double[] result = new double[d];

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }

    /**
     * Normalize a vector to unit length in-place.
     *
     * <p>If the vector is zero or has zero norm, it is left unchanged.
     */
    private static void normalize(double[] vector) {
        double norm = 0.0;
        for (double v : vector) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);

        if (norm > 0.0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }

}
