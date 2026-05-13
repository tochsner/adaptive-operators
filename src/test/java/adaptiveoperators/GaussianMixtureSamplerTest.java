package adaptiveoperators;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class GaussianMixtureSamplerTest {

    private static final double REGULARIZATION = 1e-8;
    private static final int INITIALIZATION_RECORDS = 304;

    @Test
    void initializationUsesFourSamplesWithOneHundredSkippedBetweenThem() {
        GaussianMixtureSampler sampler = new GaussianMixtureSampler(1, 1);

        for (int i = 1; i <= INITIALIZATION_RECORDS; i++) {
            sampler.record(new double[]{i}, new double[]{i * 10.0});
        }

        assertThat(sampler.mean[0][0]).isCloseTo(1.0, within(1e-12));
        assertThat(sampler.mean[0][1]).isCloseTo(10.0, within(1e-12));
        assertThat(sampler.mean[1][0]).isCloseTo(102.0, within(1e-12));
        assertThat(sampler.mean[1][1]).isCloseTo(1020.0, within(1e-12));
        assertThat(sampler.mean[2][0]).isCloseTo(203.0, within(1e-12));
        assertThat(sampler.mean[2][1]).isCloseTo(2030.0, within(1e-12));
        assertThat(sampler.mean[3][0]).isCloseTo(304.0, within(1e-12));
        assertThat(sampler.mean[3][1]).isCloseTo(3040.0, within(1e-12));
        assertThat(sampler.count).isEqualTo(INITIALIZATION_RECORDS);
        assertThat(sampler.totalEffectiveCount).isCloseTo(4.0, within(1e-12));
        for (int k = 0; k < GaussianMixtureSampler.NUM_MIXTURES; k++) {
            assertThat(sampler.effectiveCounts[k]).isCloseTo(1.0, within(1e-12));
            assertThat(sampler.covarianceSum[k][0][0]).isCloseTo(populationVariance(INITIALIZATION_RECORDS), within(1e-12));
            assertThat(sampler.covarianceSum[k][0][1]).isCloseTo(10.0 * populationVariance(INITIALIZATION_RECORDS), within(1e-10));
            assertThat(sampler.covarianceSum[k][1][1]).isCloseTo(100.0 * populationVariance(INITIALIZATION_RECORDS), within(1e-9));
        }
    }

    @Test
    void nonFiniteRecordsAreIgnored() {
        GaussianMixtureSampler sampler = new GaussianMixtureSampler(1, 1);

        sampler.record(new double[]{Double.NaN}, new double[]{1.0});
        sampler.record(new double[]{1.0}, new double[]{Double.POSITIVE_INFINITY});

        assertThat(sampler.count).isZero();
        assertThat(sampler.totalEffectiveCount).isZero();
    }

    @Test
    void onlineUpdatesKeepFinitePositiveWeightsThatSumToOne() {
        GaussianMixtureSampler sampler = new GaussianMixtureSampler(1, 1);

        for (int i = 0; i < INITIALIZATION_RECORDS + 2; i++) {
            double x = i % 20 - 10.0;
            sampler.record(new double[]{x}, new double[]{x * x});
        }

        double weightSum = 0.0;
        for (int k = 0; k < GaussianMixtureSampler.NUM_MIXTURES; k++) {
            assertThat(sampler.effectiveCounts[k]).isFinite();
            assertThat(sampler.effectiveCounts[k]).isPositive();
            weightSum += sampler.effectiveCounts[k] / sampler.totalEffectiveCount;
        }

        assertThat(sampler.count).isEqualTo(INITIALIZATION_RECORDS + 2);
        assertThat(weightSum).isCloseTo(1.0, within(1e-12));
    }

    @Test
    void sharedInitializationCovariancePreventsImmediateWeightCollapse() {
        GaussianMixtureSampler sampler = new GaussianMixtureSampler(1, 1);

        for (int i = 1; i <= INITIALIZATION_RECORDS + 1; i++) {
            sampler.record(new double[]{i}, new double[]{10.0 * i});
        }

        double maxWeight = 0.0;
        double minWeight = 1.0;
        for (int k = 0; k < GaussianMixtureSampler.NUM_MIXTURES; k++) {
            double weight = sampler.effectiveCounts[k] / sampler.totalEffectiveCount;
            maxWeight = Math.max(maxWeight, weight);
            minWeight = Math.min(minWeight, weight);
        }

        assertThat(maxWeight).isLessThan(0.8);
        assertThat(minWeight).isGreaterThan(0.01);
    }

    @Test
    void logDensityScoresConditionalMixture() {
        GaussianMixtureSampler sampler = new GaussianMixtureSampler(1, 1);
        configureConditionalMixture(sampler);

        double[] conditions = {3.0};
        double[] values = {11.5};
        double[] jointTerms = new double[GaussianMixtureSampler.NUM_MIXTURES];
        double[] conditionTerms = new double[GaussianMixtureSampler.NUM_MIXTURES];

        for (int k = 0; k < GaussianMixtureSampler.NUM_MIXTURES; k++) {
            double weight = sampler.effectiveCounts[k] / sampler.totalEffectiveCount;
            double[][] covariance = covariance(sampler, k);
            jointTerms[k] = Math.log(weight)
                    + logGaussian(new double[]{conditions[0], values[0]}, sampler.mean[k], covariance);
            conditionTerms[k] = Math.log(weight)
                    + logGaussian(
                    conditions,
                    new double[]{sampler.mean[k][0]},
                    new double[][]{{covariance[0][0]}}
            );
        }

        double expected = logSumExp(jointTerms) - logSumExp(conditionTerms);

        assertThat(sampler.logDensity(conditions, values)).isCloseTo(expected, within(1e-12));
    }

    @Test
    void conditionalSamplesUseConditionWeightedComponents() throws ReflectiveOperationException {
        GaussianMixtureSampler sampler = new GaussianMixtureSampler(1, 1);
        configureDominantConditionalMixture(sampler);
        setSamplerSeed(sampler, 5678L);

        int sampleCount = 4_000;
        double sum = 0.0;
        for (int i = 0; i < sampleCount; i++) {
            sum += sampler.sampleConditionally(new double[]{0.0})[0];
        }

        assertThat(sum / sampleCount).isCloseTo(5.0, within(0.08));
    }

    @Test
    void zeroConditionSamplerUsesUnconditionalMixture() throws ReflectiveOperationException {
        GaussianMixtureSampler sampler = new GaussianMixtureSampler(0, 1);
        configureUnconditionalMixture(sampler);
        setSamplerSeed(sampler, 9012L);

        double[] values = {0.0};
        double[] terms = new double[GaussianMixtureSampler.NUM_MIXTURES];
        for (int k = 0; k < GaussianMixtureSampler.NUM_MIXTURES; k++) {
            double weight = sampler.effectiveCounts[k] / sampler.totalEffectiveCount;
            terms[k] = Math.log(weight)
                    + logGaussian(values, sampler.mean[k], covariance(sampler, k));
        }

        double[] sample = sampler.sampleConditionally(new double[0]);

        assertThat(sampler.logDensity(new double[0], values)).isCloseTo(logSumExp(terms), within(1e-12));
        assertThat(sample).hasSize(1);
        assertThat(sample[0]).isFinite();
    }

    private static void configureConditionalMixture(GaussianMixtureSampler sampler) {
        initializeAllComponents(sampler);
        sampler.effectiveCounts = new double[]{7.0, 1.0, 1.0, 1.0};
        sampler.totalEffectiveCount = 10.0;
        sampler.mean = new double[][]{
                {0.0, 10.0},
                {20.0, -10.0},
                {-20.0, 4.0},
                {10.0, 30.0}
        };
        setCovariance(sampler, 0, new double[][]{{4.0, 2.0}, {2.0, 9.0}});
        setCovariance(sampler, 1, new double[][]{{3.0, 0.0}, {0.0, 2.0}});
        setCovariance(sampler, 2, new double[][]{{2.0, 0.0}, {0.0, 5.0}});
        setCovariance(sampler, 3, new double[][]{{6.0, 1.0}, {1.0, 3.0}});
    }

    private static void configureDominantConditionalMixture(GaussianMixtureSampler sampler) {
        initializeAllComponents(sampler);
        sampler.effectiveCounts = new double[]{7.0, 1.0, 1.0, 1.0};
        sampler.totalEffectiveCount = 10.0;
        sampler.mean = new double[][]{
                {0.0, 5.0},
                {20.0, -10.0},
                {-20.0, 15.0},
                {40.0, 25.0}
        };
        setCovariance(sampler, 0, new double[][]{{1.0, 0.0}, {0.0, 0.25}});
        setCovariance(sampler, 1, new double[][]{{1.0, 0.0}, {0.0, 1.0}});
        setCovariance(sampler, 2, new double[][]{{1.0, 0.0}, {0.0, 1.0}});
        setCovariance(sampler, 3, new double[][]{{1.0, 0.0}, {0.0, 1.0}});
    }

    private static void configureUnconditionalMixture(GaussianMixtureSampler sampler) {
        initializeAllComponents(sampler);
        sampler.effectiveCounts = new double[]{4.0, 3.0, 2.0, 1.0};
        sampler.totalEffectiveCount = 10.0;
        sampler.mean = new double[][]{{0.0}, {5.0}, {-5.0}, {10.0}};
        setCovariance(sampler, 0, new double[][]{{1.0}});
        setCovariance(sampler, 1, new double[][]{{2.0}});
        setCovariance(sampler, 2, new double[][]{{3.0}});
        setCovariance(sampler, 3, new double[][]{{4.0}});
    }

    private static void initializeAllComponents(GaussianMixtureSampler sampler) {
        for (int i = 0; i < INITIALIZATION_RECORDS; i++) {
            sampler.record(new double[sampler.numConditions], new double[sampler.numValues]);
        }
    }

    private static void setCovariance(GaussianMixtureSampler sampler, int component, double[][] covariance) {
        double count = sampler.effectiveCounts[component];
        for (int i = 0; i < covariance.length; i++) {
            for (int j = 0; j < covariance[i].length; j++) {
                sampler.covarianceSum[component][i][j] = covariance[i][j] * count;
            }
        }
    }

    private static double[][] covariance(GaussianMixtureSampler sampler, int component) {
        double[][] covariance = new double[sampler.mean[component].length][sampler.mean[component].length];
        for (int i = 0; i < covariance.length; i++) {
            for (int j = 0; j < covariance[i].length; j++) {
                covariance[i][j] = sampler.covarianceSum[component][i][j] / sampler.effectiveCounts[component];
            }
            covariance[i][i] += REGULARIZATION;
        }
        return covariance;
    }

    private static double logGaussian(double[] x, double[] mean, double[][] covariance) {
        if (x.length == 1) {
            double variance = covariance[0][0];
            double diff = x[0] - mean[0];
            return -0.5 * (Math.log(2.0 * Math.PI * variance) + diff * diff / variance);
        }

        double a = covariance[0][0];
        double b = covariance[0][1];
        double c = covariance[1][0];
        double d = covariance[1][1];
        double determinant = a * d - b * c;
        double diff0 = x[0] - mean[0];
        double diff1 = x[1] - mean[1];
        double quadratic = (d * diff0 * diff0 - b * diff1 * diff0 - c * diff0 * diff1 + a * diff1 * diff1)
                / determinant;

        return -0.5 * (2.0 * Math.log(2.0 * Math.PI) + Math.log(determinant) + quadratic);
    }

    private static double logSumExp(double[] values) {
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

    private static double populationVariance(int n) {
        return (n * n - 1.0) / 12.0;
    }

    private static void setSamplerSeed(GaussianMixtureSampler sampler, long seed)
            throws ReflectiveOperationException {
        Field rng = GaussianMixtureSampler.class.getDeclaredField("rng");
        rng.setAccessible(true);
        ((Random) rng.get(sampler)).setSeed(seed);
    }
}
