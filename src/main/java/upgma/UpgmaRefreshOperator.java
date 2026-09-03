package upgma;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.distance.JukesCantorDistance;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.spec.type.RealScalar;
import beast.base.util.Randomizer;

import java.util.List;

/**
 * Implement an operator that:
 * 1.   Samples a distance matrix D conditioned on the current tree such that UPGMA(D) = T.
 * 2.   Perturbs D by adding Gaussian noise.
 * 3.   Performs UPGMA to reconstruct the perturbed tree.
 * See @approach.md for a description of the conditional density.
 */
public class UpgmaRefreshOperator extends TreeOperator {

    public final Input<Double> concentrationInput = new Input<>("concentration", "", 10.0);
    public final Input<RealScalar<?>> substitutionRateInput = new Input<>("substitutionRate", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");

    Tree tree;
    double concentration;
    RealScalar<?> substitutionRate;
    Alignment alignment;

    double scaleFactor = 0.01;

    int n;
    double[][] substitutionDistances;
    double[] tipDates;

    double[][] currentDistances;
    double[][] proposedDistances;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.concentration = this.concentrationInput.get();
        this.substitutionRate = this.substitutionRateInput.get();
        this.alignment = this.alignmentInput.get();

        this.n = this.tree.getLeafNodeCount();
        this.currentDistances = new double[n][n];
        this.proposedDistances = new double[n][n];

        this.tipDates = this.computeTipDates();
        this.substitutionDistances = this.computeSubstitutionDistances(this.alignment);
    }

    /**
     * Computes the tip dates, indexed by tree leaf node number.
     */
    private double[] computeTipDates() {
        double[] tipDates = new double[this.n];
        for (int i = 0; i < this.n; i++) {
            tipDates[i] = this.tree.getNode(i).getHeight();
        }
        return tipDates;
    }

    /**
     * Computes the pairwise JC69 distances in expected substitutions, indexed by tree leaf node
     * number (not by alignment taxon index, which need not agree with it) so the result lines up
     * with {@link #tipDates}, {@link #currentDistances}, {@link #proposedDistances}, and the
     * BEAST-node-numbered convention {@code TreeUtils.applyClustering} expects.
     */
    private double[][] computeSubstitutionDistances(Alignment alignment) {
        JukesCantorDistance distance = new JukesCantorDistance();
        distance.setPatterns(alignment);

        double[][] substitutionDistances = new double[this.n][this.n];
        for (int i = 0; i < this.n; i++) {
            int taxonIndexI = alignment.getTaxonIndex(this.tree.getNode(i).getID());

            for (int j = i + 1; j < this.n; j++) {
                int taxonIndexJ = alignment.getTaxonIndex(this.tree.getNode(j).getID());

                double d = distance.pairwiseDistance(taxonIndexI, taxonIndexJ);
                substitutionDistances[i][j] = d;
                substitutionDistances[j][i] = d;
            }
        }

        return substitutionDistances;
    }

    @Override
    public double proposal() {
        // sample the distance matrix conditioned on T

        this.sampleDistances(this.currentDistances, this.tree);
        double oldTreeLogDensity = this.getConditionalLogDensity(this.currentDistances, this.tree);
        
        // perturb the distance matrix with a Gaussian of scale scaleFactor

        this.perturbDistances(this.currentDistances, this.proposedDistances);
        if (this.isInvalidDistanceMatrix(this.proposedDistances)) {
            return Double.NEGATIVE_INFINITY;
        }

        // reconstruct the new tree

        this.updateTree(this.tree, this.proposedDistances);
        if (this.isInvalidTree(this.tree)) {
            return Double.NEGATIVE_INFINITY;
        }

        double newTreeLogDensity = this.getConditionalLogDensity(this.proposedDistances, this.tree);

        // compute and return the HR ratio

        return newTreeLogDensity - oldTreeLogDensity;
    }

    /**
     * Samples a new distance matrix conditioned on the given tree. The distances correspond to a tree with extant
     * branches extended to the present.
     */
    private void sampleDistances(double[][] distances, Tree tree) {
        UpgmaDistanceSampler sampler = new UpgmaDistanceSampler(tree, this.substitutionDistances, this.substitutionRate, this.concentration);
        sampler.sample(distances);
    }

    /**
     * Returns the conditional density of the distances conditioned on the tree (including the Jacobian correction).
     * The distances correspond to a tree with extant branches extended to the present.
     */
    private double getConditionalLogDensity(double[][] distances, Tree tree) {
        UpgmaConditionalDensity density =
                new UpgmaConditionalDensity(tree, this.substitutionDistances, this.substitutionRate, this.concentration);
        return density.logDensity(distances);
    }

    /**
     * Perturbs currentDistances and stores the new distances in proposedDistances.
     * Every distance is perturbed by a random Gaussian with scale scaleFactor.
     */
    private void perturbDistances(double[][] currentDistances, double[][] proposedDistances) {
        for (int i = 0; i < this.n; i++) {
            for (int j = i + 1; j < this.n; j++) {
                double perturbed = currentDistances[i][j] + this.scaleFactor * Randomizer.nextGaussian();
                proposedDistances[i][j] = perturbed;
                proposedDistances[j][i] = perturbed;
            }
        }
    }

    /**
     * Updates the tree by performing UPGMA on the distance matrix. The distances correspond to a tree with
     * extant branches extended to the present, the resulting UPGMA tree gets pruned. This might lead to negative extant
     * branch lengths.
     * The node objects stay the same, the only thing that changes is the vertex relationships.
     * Tries to keep as much of the original tree intact using {@code TreeUtils.applyClustering}.
     */
    private void updateTree(Tree tree, double[][] distances) {
        this.tree.startEditing(this);

        UpgmaClustering.Result clustering = UpgmaClustering.cluster(distances);
        TreeUtils.applyClustering(tree, clustering.mergeLeft, clustering.mergeRight, clustering.mergeHeight);
    }

    /**
     * Returns true if there are negative distances.
     */
    private boolean isInvalidDistanceMatrix(double[][] distances) {
        for (int i = 0; i < this.n; i++) {
            for (int j = 0; j < this.n; j++) {
                if (distances[i][j] < 0.0) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Returns true if the tree has negative branch lengths.
     */
    private boolean isInvalidTree(Tree tree) {
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isRoot() && node.getLength() < 0.0) {
                return true;
            }
        }
        return false;
    }


    /* Scale adaption logic */

    @Override
    public double getCoercableParameterValue() {
        return this.scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0 || value > 1.0) {
            throw new IllegalArgumentException("scaleFactor must be finite and in (0, 1]");
        }
        this.scaleFactor = value;
    }

    @Override
    public void optimize(double logAlpha) {
        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.scaleFactor);
        this.scaleFactor = Math.exp(delta);
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }
}
