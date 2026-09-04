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
import java.util.SplittableRandom;

/**
 * Implement an operator that:
 * 1.   Samples a distance matrix D conditioned on the current tree such that UPGMA(D) = T.
 * 2.   Perturbs D by adding Gaussian noise to entries.
 * 3.   Performs UPGMA to reconstruct the perturbed tree.
 * See @approach.md for a description of the conditional density.
 */
public class UpgmaRefreshOperator extends TreeOperator {

    public final Input<Double> concentrationInput = new Input<>("concentration", "", 100.0);
    public final Input<Integer> affectedNodeCountInput = new Input<>("affectedCladeSize", "", 0);
    public final Input<String> selectionModeInput = new Input<>("selectionMode", "which distances get perturbed: uniform, all or clade", "clade");
    public final Input<RealScalar<?>> substitutionRateInput = new Input<>("substitutionRate", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");

    Tree tree;
    double concentration;
    int affectedNodeCount;
    RealScalar<?> substitutionRate;
    Alignment alignment;

    double scaleFactor = 0.1;

    int n;
    double[][] substitutionDistances;
    double[] tipDates;

    double[][] currentDistances;
    double[][] proposedDistances;

    UpgmaTaxonSelection selection;

    SplittableRandom rng;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.concentration = this.concentrationInput.get();
        this.affectedNodeCount = this.affectedNodeCountInput.get() > 0
                ? Math.min(this.affectedNodeCountInput.get(), this.tree.getLeafNodeCount())
                : this.tree.getLeafNodeCount();
        this.substitutionRate = this.substitutionRateInput.get();
        this.alignment = this.alignmentInput.get();

        this.n = this.tree.getLeafNodeCount();
        this.currentDistances = new double[n][n];
        this.proposedDistances = new double[n][n];
        this.selection = new UpgmaTaxonSelection(UpgmaTaxonSelection.Mode.parse(this.selectionModeInput.get()), this.tree, this.affectedNodeCount);

        this.rng = new SplittableRandom(Randomizer.nextLong());
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
     * number.
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
        
        // perturb the distance matrix with

        this.perturbDistances(this.currentDistances, this.proposedDistances);
        if (this.isInvalidDistanceMatrix(this.proposedDistances)) {
            return Double.NEGATIVE_INFINITY;
        }

        // reconstruct the new tree

        UpgmaClustering.Result newClustering = UpgmaClustering.cluster(this.proposedDistances);

        // in clade mode this rejects proposals whose clade dissolved and restores the exact heights
        // of every merge the perturbation could not have changed
        double logSelectionRatio = this.selection.reconcile(newClustering);
        if (logSelectionRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        this.updateTree(this.tree, newClustering);
        if (this.isInvalidTree(this.tree)) {
            return Double.NEGATIVE_INFINITY;
        }

        double newTreeLogDensity = this.getConditionalLogDensity(this.proposedDistances, this.tree);

        // compute and return the HR ratio

        return newTreeLogDensity - oldTreeLogDensity + logSelectionRatio;
    }

    /**
     * Returns the Monte Carlo mean of {@link #sampleDistances}, entrywise, over a hundred independent
     * draws conditioned on the current tree. Diagnostic only: at this operator's own scale, the mean
     * distance matrix has no particular meaning (it is not itself consistent with the tree under
     * UPGMA), but it is a cheap way to sanity-check that the sampler's draws are centered where the
     * merge distances say they should be.
     */
    private double[][] getMeanDistance() {
        double[][] mean = new double[this.n][this.n];
        double[][] sample = new double[this.n][this.n];

        int draws = 100;
        for (int k = 0; k < draws; k++) {
            this.sampleDistances(sample, this.tree);
            for (int i = 0; i < this.n; i++) {
                for (int j = 0; j < this.n; j++) {
                    mean[i][j] += sample[i][j];
                }
            }

            System.out.println(sample[1][2] + "\t" + sample[1][3] + "\t" + sample[1][4] + "\t|\t" + sample[2][3] + "\t" + sample[3][4] +  "\t|\t" + sample[3][4]);
        }

        for (int i = 0; i < this.n; i++) {
            for (int j = 0; j < this.n; j++) {
                mean[i][j] /= draws;
            }
        }

        System.out.println("-------------------------------------------------------");
        System.out.println(mean[1][2] + "\t" + mean[1][3] + "\t" + mean[1][4] + "\t|\t" + mean[2][3] + "\t" + mean[2][4] +  "\t|\t" + mean[3][4]);
        System.out.println(substitutionDistances[1][2] + "\t" + substitutionDistances[1][3] + "\t" + substitutionDistances[1][4] + "\t|\t" + substitutionDistances[2][3] + "\t" + substitutionDistances[2][4] +  "\t|\t" + substitutionDistances[3][4]);

        return mean;
    }

    /**
     * Samples a new distance matrix conditioned on the given tree. The distances correspond to a tree with extant
     * branches extended to the present.
     */
    private void sampleDistances(double[][] distances, Tree tree) {
        UpgmaDistanceSampler sampler = new UpgmaDistanceSampler(tree, this.substitutionDistances, this.tipDates, this.substitutionRate, this.concentration, this.rng);
        sampler.sample(distances);
    }

    /**
     * Returns the conditional density of the distances conditioned on the tree (including the Jacobian correction).
     * The distances correspond to a tree with extant branches extended to the present.
     */
    private double getConditionalLogDensity(double[][] distances, Tree tree) {
        UpgmaConditionalDensity density = new UpgmaConditionalDensity(tree, this.substitutionDistances, this.tipDates, this.substitutionRate, this.concentration);
        return density.logDensity(distances);
    }

    /**
     * Perturbs currentDistances and stores the new distances in proposedDistances.
     * Every distance between two taxa in the affected set, drawn by {@link #selection}, gets
     * {@code scaleFactor * N(0, 1)} added; every other distance is copied through unchanged. The
     * kernel is symmetric, so no Hastings correction is needed. A perturbed distance can go
     * negative.
     */
    private void perturbDistances(double[][] currentDistances, double[][] proposedDistances) {
        this.selection.select();

        for (int i = 0; i < this.n; i++) {
            boolean affectedI = this.selection.isAffected(i);
            for (int j = i + 1; j < this.n; j++) {
                double value = currentDistances[i][j];
                if (affectedI && this.selection.isAffected(j)) {
                    value += this.scaleFactor * this.rng.nextGaussian();
                }
                proposedDistances[i][j] = value;
                proposedDistances[j][i] = value;
            }
        }
    }

    /**
     * Runs UPGMA clustering of the distance matrix. The distances correspond to a tree with
     * extant branches extended to the present, the resulting UPGMA tree gets pruned.
     * This might lead to negative extant branch lengths.
     */
    private void updateTree(Tree tree, UpgmaClustering.Result clustering) {
        this.tree.startEditing(this);
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
