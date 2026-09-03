package distancematrix;

import beast.base.util.Randomizer;
import org.apache.commons.statistics.distribution.LogNormalDistribution;

/**
 * Log-normal reference density on the matrix entries.
 *
 * <p>Heavier tailed than the exponential, which lets entries far above the tree they induce sit
 * comfortably instead of being pulled down hard.
 */
public class LogNormalReferenceDensity implements ReferenceDensity {

    private final LogNormalDistribution distribution;

    public LogNormalReferenceDensity(double logMean, double logStandardDeviation) {
        if (!(logStandardDeviation > 0.0) || !Double.isFinite(logStandardDeviation)) {
            throw new IllegalArgumentException(
                    "the log standard deviation must be finite and positive");
        }

        this.distribution = LogNormalDistribution.of(logMean, logStandardDeviation);
    }

    @Override
    public double logDensity(double value) {
        if (!(value > 0.0)) {
            return Double.NEGATIVE_INFINITY;
        }

        return this.distribution.logDensity(value);
    }

    @Override
    public double logSurvival(double value) {
        if (!(value > 0.0)) {
            return 0.0;
        }

        return Math.log(this.distribution.survivalProbability(value));
    }

    @Override
    public double sampleAbove(double threshold) {
        double survival = this.distribution.survivalProbability(Math.max(threshold, 0.0));

        if (!(survival > 0.0)) {
            // the threshold sits beyond everything the tail can represent
            return threshold;
        }

        return Math.max(
                this.distribution.inverseSurvivalProbability(survival * Randomizer.nextDouble()),
                threshold);
    }
}
