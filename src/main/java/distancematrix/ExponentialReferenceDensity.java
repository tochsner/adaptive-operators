package distancematrix;

import beast.base.util.Randomizer;

/**
 * Exponential reference density on the matrix entries.
 *
 * <p>The survival function is available in closed form, which keeps the fibre volume cheap: the
 * whole correction reduces to a linear function of the node distances plus a log density term.
 */
public class ExponentialReferenceDensity implements ReferenceDensity {

    private final double rate;
    private final double logRate;

    public ExponentialReferenceDensity(double rate) {
        if (!(rate > 0.0) || !Double.isFinite(rate)) {
            throw new IllegalArgumentException("the exponential rate must be finite and positive");
        }

        this.rate = rate;
        this.logRate = Math.log(rate);
    }

    @Override
    public double logDensity(double value) {
        if (!(value > 0.0)) {
            return Double.NEGATIVE_INFINITY;
        }

        return this.logRate - this.rate * value;
    }

    @Override
    public double logSurvival(double value) {
        if (!(value > 0.0)) {
            return 0.0;
        }

        return -this.rate * value;
    }

    @Override
    public double sampleAbove(double threshold) {
        // the exponential is memoryless, so the excess over the threshold is exponential again
        return threshold - Math.log(Randomizer.nextDouble()) / this.rate;
    }
}
