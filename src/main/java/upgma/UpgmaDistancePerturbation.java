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
        GAUSSIAN,
        /**
         * The AR(1) step {@code d' = mu + sqrt(1 - beta^2) (d - mu) + beta sigma eps} towards the
         * per-pair surrogate {@code N(mu, sigma^2)}, see ar1-perturbation.md. The scale factor is
         * {@code beta} in {@code (0, 1]}.
         */
        AR1;

        /** Parses a case-insensitive mode name. */
        public static Mode parse(String name) {
            try {
                return Mode.valueOf(name.trim().toUpperCase());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("perturbationMode must be gaussian or ar1, got: " + name);
            }
        }

        /** Whether the scale factor is a fraction in {@code (0, 1]}. */
        public boolean isBounded() {
            return this == AR1;
        }
    }

    private final Mode mode;
    private final UpgmaTaxonSelection selection;
    private final SplittableRandom rng;

    private double scaleFactor = 0.1;

    // the AR1 surrogate in substitution units, rescaled into D's extended time units with an
    // adaptive rate that is held fixed within a proposal
    private double[][] substitutionDistances;
    private double[][] substitutionVariances;
    private double[] tipDates;

    private double logRate = Double.NaN;
    private double observedLogRate = Double.NaN;
    private long observationCount;

    public UpgmaDistancePerturbation(Mode mode, UpgmaTaxonSelection selection, SplittableRandom rng) {
        this.mode = mode;
        this.selection = selection;
        this.rng = rng;
    }

    /**
     * Sets the {@link Mode#AR1} surrogate, given in substitution units. In D's units it is
     * {@code N(delta_ij / rate + t_i + t_j, v_ij / rate^2)} with the adaptive {@link #getRate rate}.
     */
    public void setSurrogate(double[][] substitutionDistances, double[][] substitutionVariances, double[] tipDates) {
        this.substitutionDistances = substitutionDistances;
        this.substitutionVariances = substitutionVariances;
        this.tipDates = tipDates;
    }

    /**
     * Draws a new affected set, writes the perturbed distances into {@code proposedDistances} and
     * returns the log Hastings correction, {@code -inf} if the move has to be rejected. A perturbed
     * distance can go negative.
     */
    public double perturb(double[][] currentDistances, double[][] proposedDistances) {
        this.selection.select();

        return switch (this.mode) {
            case GAUSSIAN -> this.perturbGaussian(currentDistances, proposedDistances);
            case AR1 -> this.perturbAr1(currentDistances, proposedDistances);
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

    /**
     * Applies the AR(1) step to every affected distance with the rate held fixed, so the kernel is
     * reversible with respect to the surrogate {@code f} and the correction is
     * {@code log f(d) - log f(d')} summed over pairs. The rate the current matrix implies is
     * recorded for {@link #adaptRate}.
     */
    private double perturbAr1(double[][] currentDistances, double[][] proposedDistances) {
        if (this.substitutionDistances == null) {
            throw new IllegalStateException("the ar1 perturbation needs a surrogate, call setSurrogate first");
        }

        this.observeRate(currentDistances);
        if (Double.isNaN(this.logRate)) {
            return Double.NEGATIVE_INFINITY;
        }
        double rate = Math.exp(this.logRate);

        double beta = this.scaleFactor;
        double shrink = Math.sqrt(1.0 - beta * beta);
        double logRatio = 0.0;

        for (int i = 0; i < currentDistances.length; i++) {
            boolean affectedI = this.selection.isAffected(i);
            for (int j = i + 1; j < currentDistances.length; j++) {
                double value = currentDistances[i][j];
                if (affectedI && this.selection.isAffected(j)) {
                    double mean = this.substitutionDistances[i][j] / rate + this.tipDates[i] + this.tipDates[j];
                    double variance = this.substitutionVariances[i][j] / (rate * rate);

                    double current = value - mean;
                    double proposed = shrink * current + beta * Math.sqrt(variance) * this.rng.nextGaussian();
                    logRatio += (proposed * proposed - current * current) / (2.0 * variance);
                    double newValue = mean + proposed;
                    value = newValue;
                }
                proposedDistances[i][j] = value;
                proposedDistances[j][i] = value;
            }
        }

        return logRatio;
    }

    /**
     * Records the rate {@code distances} implies, the summed substitution distance over the summed
     * path length of every pair, and seeds the adaptive rate with it if there is none yet.
     */
    private void observeRate(double[][] distances) {
        double substitutions = 0.0;
        double time = 0.0;
        for (int i = 0; i < distances.length; i++) {
            for (int j = i + 1; j < distances.length; j++) {
                substitutions += this.substitutionDistances[i][j];
                time += distances[i][j] - this.tipDates[i] - this.tipDates[j];
            }
        }

        this.observedLogRate = time > 0.0 && substitutions > 0.0 ? Math.log(substitutions / time) : Double.NaN;
        if (Double.isNaN(this.logRate) && !Double.isNaN(this.observedLogRate)) {
            this.logRate = this.observedLogRate;
            this.observedLogRate = Double.NaN;
            this.observationCount = 1;
        }
    }

    /**
     * Folds the rate observed in the last proposal into the running log-scale mean with step
     * {@code 1 / k}. Meant to be called from the operator's {@code optimize}, so the rate only
     * moves between proposals under BEAST's adaptation schedule.
     */
    public void adaptRate() {
        if (Double.isNaN(this.observedLogRate)) {
            return;
        }
        this.observationCount++;
        this.logRate += (this.observedLogRate - this.logRate) / this.observationCount;
        this.observedLogRate = Double.NaN;
    }

    /** The rate the {@link Mode#AR1} surrogate is currently scaled with, {@code NaN} before the first proposal. */
    public double getRate() {
        return Math.exp(this.logRate);
    }

    public Mode getMode() {
        return this.mode;
    }

    public double getScaleFactor() {
        return this.scaleFactor;
    }

    /** Clamped into {@code (0, 1]} for a bounded {@link Mode}. */
    public void setScaleFactor(double scaleFactor) {
        if (this.mode.isBounded()) {
            scaleFactor = Math.min(1.0, Math.max(1e-6, scaleFactor));
        }
        this.scaleFactor = scaleFactor;
    }
}
