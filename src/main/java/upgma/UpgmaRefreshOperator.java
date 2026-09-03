package upgma;

import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.distance.JukesCantorDistance;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Tree;

/**
 * Implement an operator that:
 * 1.   Samples a distance matrix D conditioned on the current tree such that UPGMA(D) = T.
 * 2.   Perturbs D by adding Gaussian noise.
 * 3.   Performs UPGMA to reconstruct the perturbed tree.
 * See @approach.md for a description of the conditional density.
 */
public class UpgmaRefreshOperator extends TreeOperator {

    public final Input<Double> concentrationInput = new Input<>("concentration", "", 1.0);
    public final Input<Double> substitutionRateInput = new Input<>("substitutionRate", "", 1.0);
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");

    Tree tree;
    double concentration;
    double substitutionRate;
    Alignment alignment;

    double scaleFactor = 1.0;

    int n;
    double[][] substitutionDistances;

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

        this.substitutionDistances = this.computeSubstitutionDistances(this.alignment);
    }

    /**
     * Computes the pairwise JC69 distances in expected substitutions.
     */
    private double[][] computeSubstitutionDistances(Alignment alignment) {
        JukesCantorDistance distance = new JukesCantorDistance();
        distance.setPatterns(alignment);

        int n = alignment.getSiteCount();
        double[][] substitutionDistances = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = distance.pairwiseDistance(i, j);
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

        // reconstruct the new tree

        this.updateTree(this.tree, this.proposedDistances);
        double newTreeLogDensity = this.getConditionalLogDensity(this.proposedDistances, this.tree);

        // compute and return the HR ratio

        return newTreeLogDensity - oldTreeLogDensity;
    }

    /**
     * Samples a new distance matrix conditioned on the given tree.
     */
    private void sampleDistances(double[][] distances, Tree tree) {
    }

    /**
     * Returns the conditional density of the distances conditioned on the tree.
     */
    private double getConditionalLogDensity(double[][] distances, Tree tree) {
    }

    /**
     * Perturbs currentDistances and stores the new distances in proposedDistances.
     * Every distance is perturbed by a random Gaussian with scale scaleFactor.
     */
    private void perturbDistances(double[][] currentDistances, double[][] proposedDistances) {
    }

    /**
     * Updates the tree by performing UPGMA on the distance matrix. The node objects stay the same,
     * the only thing that changes is the vertex relationships. Tries to keep as much of the original
     * tree intact.
     */
    private void updateTree(Tree tree, double[][] distances) {
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

}
