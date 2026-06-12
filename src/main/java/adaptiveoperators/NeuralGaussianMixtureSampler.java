package adaptiveoperators;

import ml.MixtureDensityNetwork;
import ml.MixtureDensityNetwork.MixtureParameters;

import java.util.Arrays;
import java.util.Random;

/**
 * Conditional sampler backed by a mixture density network that learns the mixture
 * weights and component parameters of the conditional distribution on the fly. Recorded
 * samples are standardized with running statistics before being passed to the network,
 * and samples and densities are transformed back accordingly.
 */
public class NeuralGaussianMixtureSampler extends ConditionalSampler implements AutoCloseable {

    static final int NUM_COMPONENTS = 4;
    private static final double MIN_STANDARD_DEVIATION = 1e-6;

    private final MixtureDensityNetwork network;
    private final Random rng = new Random();

    // running standardization statistics over [conditions, values], updated with Welford
    private final double[] statisticsMean;
    private final double[] statisticsM2;
    private long statisticsCount = 0;

    public NeuralGaussianMixtureSampler(int numConditions, int numValues) {
        super(numConditions, numValues);

        this.network = new MixtureDensityNetwork(numConditions, numValues, NUM_COMPONENTS);
        this.statisticsMean = new double[numConditions + numValues];
        this.statisticsM2 = new double[numConditions + numValues];
    }

    @Override
    public void record(double[] conditions, double[] values) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) return;
        if (!Arrays.stream(values).allMatch(Double::isFinite)) return;

        updateStatistics(conditions, values);
        this.network.record(standardizeConditions(conditions), standardizeValues(values));
    }

    @Override
    public double[] sampleConditionally(double[] conditions, double scaleFactor) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite conditions found.");
        }

        MixtureParameters parameters = this.network.predict(standardizeConditions(conditions));
        int component = sampleComponent(parameters.weights);

        double scale = Math.sqrt(scaleFactor);
        double[] values = new double[this.numValues];
        for (int i = 0; i < this.numValues; i++) {
            double standardized = parameters.means[component][i]
                    + scale * parameters.sigmas[component][i] * this.rng.nextGaussian();
            values[i] = this.statisticsMean[this.numConditions + i]
                    + standardDeviation(this.numConditions + i) * standardized;
        }
        return values;
    }

    @Override
    public double logDensity(double[] conditions, double[] values, double scaleFactor) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite conditions found.");
        }
        if (!Arrays.stream(values).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite values found.");
        }

        MixtureParameters parameters = this.network.predict(standardizeConditions(conditions));
        double[] standardizedValues = standardizeValues(values);

        double[] componentTerms = new double[NUM_COMPONENTS];
        for (int k = 0; k < NUM_COMPONENTS; k++) {
            double logDensity = Math.log(parameters.weights[k]);
            for (int i = 0; i < this.numValues; i++) {
                double sigma = Math.sqrt(scaleFactor) * parameters.sigmas[k][i];
                double diff = (standardizedValues[i] - parameters.means[k][i]) / sigma;
                logDensity += -0.5 * (Math.log(2.0 * Math.PI) + diff * diff) - Math.log(sigma);
            }
            componentTerms[k] = logDensity;
        }

        // the Jacobian of the standardization accounts for the change of variables
        double logJacobian = 0.0;
        for (int i = 0; i < this.numValues; i++) {
            logJacobian -= Math.log(standardDeviation(this.numConditions + i));
        }

        return logSumExp(componentTerms) + logJacobian;
    }

    private void updateStatistics(double[] conditions, double[] values) {
        this.statisticsCount += 1;
        for (int i = 0; i < this.statisticsMean.length; i++) {
            double x = i < this.numConditions ? conditions[i] : values[i - this.numConditions];
            double delta = x - this.statisticsMean[i];
            this.statisticsMean[i] += delta / this.statisticsCount;
            this.statisticsM2[i] += delta * (x - this.statisticsMean[i]);
        }
    }

    private double standardDeviation(int index) {
        if (this.statisticsCount < 2) return 1.0;
        return Math.max(
                Math.sqrt(this.statisticsM2[index] / (this.statisticsCount - 1)),
                MIN_STANDARD_DEVIATION
        );
    }

    private double[] standardizeConditions(double[] conditions) {
        double[] standardized = new double[this.numConditions];
        for (int i = 0; i < this.numConditions; i++) {
            standardized[i] = (conditions[i] - this.statisticsMean[i]) / standardDeviation(i);
        }
        return standardized;
    }

    private double[] standardizeValues(double[] values) {
        double[] standardized = new double[this.numValues];
        for (int i = 0; i < this.numValues; i++) {
            int index = this.numConditions + i;
            standardized[i] = (values[i] - this.statisticsMean[index]) / standardDeviation(index);
        }
        return standardized;
    }

    private int sampleComponent(double[] weights) {
        double u = this.rng.nextDouble();
        double cumulative = 0.0;
        for (int k = 0; k < weights.length; k++) {
            cumulative += weights[k];
            if (u <= cumulative) return k;
        }
        return weights.length - 1;
    }

    private double logSumExp(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (value > max) max = value;
        }

        double sum = 0.0;
        for (double value : values) {
            sum += Math.exp(value - max);
        }
        return max + Math.log(sum);
    }

    @Override
    public void close() {
        this.network.close();
    }
}
