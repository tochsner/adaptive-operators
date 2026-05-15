package adaptiveoperators;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class MultivariateNormalSamplerTest {

    @Test
    void repeatedConditionalSamplesHaveExpectedMeanAndCovariance() throws ReflectiveOperationException {
        MultivariateNormalSampler sampler = new MultivariateNormalSampler(1, 2);
        sampler.count = 1;
        sampler.mean = new double[]{1.0, 2.0, -1.0};
        sampler.covarianceSum = new double[][]{
                {4.0, 2.0, -1.0},
                {2.0, 9.0, 3.0},
                {-1.0, 3.0, 16.0}
        };
        setSamplerSeed(sampler, 5678L);

        int sampleCount = 10_000;
        double[] conditions = {3.0};
        double[][] samples = new double[sampleCount][];
        for (int i = 0; i < sampleCount; i++) {
            samples[i] = sampler.sampleConditionally(conditions);
        }

        double[] sampleMean = batchMean(samples);
        double[][] sampleCovariance = covariance(samples, sampleMean);
        double[] expectedMean = {3.0, -1.5};
        double[][] expectedCovariance = {
                {8.0, 3.5},
                {3.5, 15.75}
        };

        for (int i = 0; i < expectedMean.length; i++) {
            assertThat(sampleMean[i]).isCloseTo(expectedMean[i], within(0.15));
        }
        for (int i = 0; i < expectedCovariance.length; i++) {
            for (int j = 0; j < expectedCovariance[i].length; j++) {
                assertThat(sampleCovariance[i][j]).isCloseTo(expectedCovariance[i][j], within(0.6));
            }
        }
    }

    @Test
    void logDensityScoresValuesConditionedOnConditions() {
        MultivariateNormalSampler sampler = new MultivariateNormalSampler(1, 2);
        sampler.count = 1;
        sampler.mean = new double[]{1.0, 2.0, -1.0};
        sampler.covarianceSum = new double[][]{
                {4.0, 2.0, -1.0},
                {2.0, 9.0, 3.0},
                {-1.0, 3.0, 16.0}
        };

        double[] conditions = {3.0};
        double[] valuesAtConditionalMean = {3.0, -1.5};
        double determinant = 8.0 * 15.75 - 3.5 * 3.5;
        double expected = -0.5 * (2.0 * Math.log(2.0 * Math.PI) + Math.log(determinant));

        assertThat(sampler.logDensity(conditions, valuesAtConditionalMean)).isCloseTo(expected, within(1e-12));
    }

    private static double[] batchMean(double[][] observations) {
        double[] mean = new double[observations[0].length];
        for (double[] observation : observations) {
            for (int i = 0; i < observation.length; i++) {
                mean[i] += observation[i];
            }
        }
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= observations.length;
        }
        return mean;
    }

    private static double[][] batchM2(double[][] observations, double[] mean) {
        double[][] m2 = new double[mean.length][mean.length];
        for (double[] observation : observations) {
            for (int i = 0; i < mean.length; i++) {
                for (int j = 0; j < mean.length; j++) {
                    m2[i][j] += (observation[i] - mean[i]) * (observation[j] - mean[j]);
                }
            }
        }
        return m2;
    }

    private static double[][] covariance(double[][] observations, double[] mean) {
        double[][] covariance = batchM2(observations, mean);
        for (int i = 0; i < covariance.length; i++) {
            for (int j = 0; j < covariance[i].length; j++) {
                covariance[i][j] /= observations.length;
            }
        }
        return covariance;
    }

    private static void setSamplerSeed(MultivariateNormalSampler sampler, long seed)
            throws ReflectiveOperationException {
        Field rng = MultivariateNormalSampler.class.getDeclaredField("rng");
        rng.setAccessible(true);
        ((Random) rng.get(sampler)).setSeed(seed);
    }

}
