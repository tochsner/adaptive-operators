package adaptiveoperators;

import java.util.Random;

/**
 * Models each taxon pair distance independently as a shifted log-normal distribution.
 * The mean and variance of the log-distance are estimated online with Welford's
 * algorithm.
 */
public class LogNormalModel implements TaxaDistanceModel {

    private static final double MIN_LOG_VARIANCE = 1e-12;

    private final double[] offset;
    private final int[] count;
    private final double[] meanLogDistance;
    private final double[] m2LogDistance;

    public LogNormalModel(double[] offsets) {
        for (double offset : offsets) {
            if (!Double.isFinite(offset) || offset < 0.0) {
                throw new IllegalArgumentException("Log-normal offset must be a finite non-negative value");
            }
        }

        this.offset = offsets.clone();
        this.count = new int[offsets.length];
        this.meanLogDistance = new double[offsets.length];
        this.m2LogDistance = new double[offsets.length];
    }

    @Override
    public void record(double[] distances, int pairIndex) {
        // the pairs are independent, so every distance contributes to its own model
        for (int i = 0; i < distances.length; i++) {
            double shiftedDistance = distances[i] - this.offset[i];
            if (!Double.isFinite(shiftedDistance) || shiftedDistance <= 0.0) {
                continue;
            }

            double logDistance = Math.log(shiftedDistance);
            this.count[i]++;
            double delta = logDistance - this.meanLogDistance[i];
            this.meanLogDistance[i] += delta / this.count[i];
            double delta2 = logDistance - this.meanLogDistance[i];
            this.m2LogDistance[i] += delta * delta2;
        }
    }

    @Override
    public double sample(double[] distances, int pairIndex, Random random, double scaleFactor) {
        double standardDeviation = Math.sqrt(getVariance(pairIndex, scaleFactor));
        return this.offset[pairIndex]
                + Math.exp(this.meanLogDistance[pairIndex] + standardDeviation * random.nextGaussian());
    }

    @Override
    public double logDensity(double[] distances, int pairIndex, double value, double scaleFactor) {
        double shiftedDistance = value - this.offset[pairIndex];
        if (!Double.isFinite(shiftedDistance) || shiftedDistance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        double variance = getVariance(pairIndex, scaleFactor);
        double logDistance = Math.log(shiftedDistance);
        double diff = logDistance - this.meanLogDistance[pairIndex];
        return -Math.log(shiftedDistance)
                - 0.5 * (Math.log(2.0 * Math.PI * variance) + diff * diff / variance);
    }

    private double getVariance(int pairIndex, double scaleFactor) {
        if (this.count[pairIndex] < 2) {
            throw new IllegalStateException("At least two distances are required to sample a log-normal model");
        }

        return scaleFactor * Math.max(this.m2LogDistance[pairIndex] / (this.count[pairIndex] - 1), MIN_LOG_VARIANCE);
    }
}
