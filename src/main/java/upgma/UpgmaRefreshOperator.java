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
 * Refreshes the tree by sampling a distance matrix D with {@code UPGMA(D) = T}, perturbing D with
 * Gaussian noise and reclustering it. See approach.md for the conditional density.
 */
public class UpgmaRefreshOperator extends TreeOperator {

    public final Input<Double> concentrationInput = new Input<>("concentration", "", 100.0);
    public final Input<Integer> affectedCladeSizeInput = new Input<>("affectedCladeSize", "", 0);
    public final Input<String> selectionModeInput = new Input<>("selectionMode", "which distances get perturbed: uniform, all or clade", "clade");
    public final Input<RealScalar<?>> substitutionRateInput = new Input<>("substitutionRate", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");

    Tree tree;
    double concentration;
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
        this.substitutionRate = this.substitutionRateInput.get();
        this.alignment = this.alignmentInput.get();

        this.n = this.tree.getLeafNodeCount();
        this.currentDistances = new double[this.n][this.n];
        this.proposedDistances = new double[this.n][this.n];
        this.selection = new UpgmaTaxonSelection(
                UpgmaTaxonSelection.Mode.parse(this.selectionModeInput.get()), this.tree,
                this.affectedCladeSizeInput.get());

        this.rng = new SplittableRandom(Randomizer.nextLong());
        this.tipDates = this.computeTipDates();
        this.substitutionDistances = this.computeSubstitutionDistances(this.alignment);
    }

    /** Computes the tip dates, indexed by tree leaf node number. */
    private double[] computeTipDates() {
        double[] tipDates = new double[this.n];
        for (int i = 0; i < this.n; i++) {
            tipDates[i] = this.tree.getNode(i).getHeight();
        }
        return tipDates;
    }

    /** Computes the pairwise JC69 distances, indexed by tree leaf node number. */
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
        // sample a distance matrix conditioned on the current tree

        UpgmaSplitModel currentModel = this.createSplitModel();
        new UpgmaDistanceSampler(currentModel, this.rng).sample(this.currentDistances);
        double currentLogDensity = new UpgmaConditionalDensity(currentModel).logDensity(this.currentDistances);

        // perturb the distance matrix

        this.perturbDistances(this.currentDistances, this.proposedDistances);
        if (hasNegativeDistance(this.proposedDistances)) {
            return Double.NEGATIVE_INFINITY;
        }

        // reconstruct the tree from the perturbed distances

        UpgmaClustering.Result clustering = UpgmaClustering.cluster(this.proposedDistances);

        // in clade mode this rejects proposals whose clade dissolved and restores the exact heights
        // of every merge the perturbation could not have changed
        double logSelectionRatio = this.selection.reconcile(clustering);
        if (logSelectionRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        this.tree.startEditing(this);
        TreeUtils.applyClustering(this.tree, clustering.mergeLeft(), clustering.mergeRight(), clustering.mergeHeight());
        if (hasNegativeBranch(this.tree)) {
            return Double.NEGATIVE_INFINITY;
        }

        // the reverse move has to draw the proposed distances from the proposed tree

        double proposedLogDensity =
                new UpgmaConditionalDensity(this.createSplitModel()).logDensity(this.proposedDistances);

        return proposedLogDensity - currentLogDensity + logSelectionRatio;
    }

    /**
     * Returns a snapshot of the current tree and substitution rate for the sampler and the density.
     * The distances it describes correspond to a tree with extant branches extended to the present.
     */
    private UpgmaSplitModel createSplitModel() {
        return new UpgmaSplitModel(
                this.tree, this.substitutionDistances, this.tipDates, this.substitutionRate, this.concentration);
    }

    /**
     * Adds {@code scaleFactor * N(0, 1)} to every distance between two taxa in the affected set and
     * copies the rest through unchanged. The kernel is symmetric, so it needs no Hastings
     * correction, but a perturbed distance can go negative.
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

    /** Returns true if any distance is negative, which UPGMA cannot interpret. */
    private static boolean hasNegativeDistance(double[][] distances) {
        for (double[] row : distances) {
            for (double d : row) {
                if (d < 0.0) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Returns true if the tree has a negative branch length, which pruning the extended extant
     * branches back to the sampling times can produce.
     */
    private static boolean hasNegativeBranch(Tree tree) {
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
