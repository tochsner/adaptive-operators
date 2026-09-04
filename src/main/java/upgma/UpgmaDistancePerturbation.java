package upgma;

import java.util.SplittableRandom;

/**
 * Perturbs the sampled distance matrix in a {@link UpgmaRefreshOperator} proposal, which is what
 * moves the tree: the perturbed matrix is reclustered into the proposed tree.
 *
 * <p>Only the distances between taxa the {@link UpgmaTaxonSelection} picked are touched, and every
 * other distance is copied through unchanged. {@link #perturb} returns the log Hastings correction
 * of the kernel its {@link Mode} applied, which is {@code 0} for the symmetric random walk.
 */
public final class UpgmaDistancePerturbation {

    /** How the selected distances are perturbed. */
    public enum Mode {
        /** A symmetric random walk: {@code scaleFactor * N(0, 1)} added to every selected distance. */
        GAUSSIAN;

        /** Parses a case-insensitive mode name. */
        public static Mode parse(String name) {
            try {
                return Mode.valueOf(name.trim().toUpperCase());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("perturbationMode must be gaussian, got: " + name);
            }
        }
    }

    private final Mode mode;
    private final UpgmaTaxonSelection selection;
    private final SplittableRandom rng;

    private double scaleFactor = 0.1;

    public UpgmaDistancePerturbation(Mode mode, UpgmaTaxonSelection selection, SplittableRandom rng) {
        this.mode = mode;
        this.selection = selection;
        this.rng = rng;
    }

    /**
     * Draws a new affected set, writes the perturbed distances into {@code proposedDistances} and
     * returns the log Hastings correction of the kernel. A perturbed distance can go negative.
     */
    public double perturb(double[][] currentDistances, double[][] proposedDistances) {
        this.selection.select();

        return switch (this.mode) {
            case GAUSSIAN -> this.perturbGaussian(currentDistances, proposedDistances);
        };
    }

    /** Adds {@code scaleFactor * N(0, 1)} to every affected distance. */
    private double perturbGaussian(double[][] currentDistances, double[][] proposedDistances) {
        for (int i = 0; i < currentDistances.length; i++) {
            boolean affectedI = this.selection.isAffected(i);
            for (int j = i + 1; j < currentDistances.length; j++) {
                double value = currentDistances[i][j];
                if (affectedI && this.selection.isAffected(j)) {
                    value += this.scaleFactor * this.rng.nextGaussian();
                }
                proposedDistances[i][j] = value;
                proposedDistances[j][i] = value;
            }
        }

        // the random walk is symmetric, so it needs no correction
        return 0.0;
    }

    public double getScaleFactor() {
        return this.scaleFactor;
    }

    public void setScaleFactor(double scaleFactor) {
        this.scaleFactor = scaleFactor;
    }
}
