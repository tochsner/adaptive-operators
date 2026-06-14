package mala;

import adapters.Adapter;
import adaptiveoperators.CenteredMultivariateNormalSampler;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class LargeJumpMALAOperatorTest {

    @Test
    public void testOptimizationImprovesLikelihood() {
        int dimension = 3;
        FakeAdapter adapter = new FakeAdapter(dimension);
        LargeJumpMALAOperator operator = newOperator(adapter, dimension, 30);

        // likelihood peaks at the origin
        Supplier<Double> logLikelihood = () -> negativeSquaredNorm(adapter.values);

        double[] start = {3.0, -3.0, 2.5};

        double[] optimized = operator.jumpAndOptimize(new double[0], start, 42L, 0, logLikelihood, state);

        assertThat(optimized).isNotNull();

        // hill-climbing keeps only improving moves, so the result is at least as good as the start
        assertThat(negativeSquaredNorm(optimized)).isGreaterThanOrEqualTo(negativeSquaredNorm(start));
    }

    @Test
    public void testOptimizationIsDeterministicGivenSeed() {
        int dimension = 4;
        FakeAdapter adapter = new FakeAdapter(dimension);
        LargeJumpMALAOperator operator = newOperator(adapter, dimension, 25);

        Supplier<Double> logLikelihood = () -> negativeSquaredNorm(adapter.values);

        double[] start = {1.0, -1.0, 1.0, -1.0};

        // proposal() draws the auxiliaries before the walks, which creates this thread's RNG;
        // do the same here so both setSeed(7) calls reseed the same pre-existing RNG
        Randomizer.nextLong();

        double[] first = operator.jumpAndOptimize(new double[0], start, 7L, 0, logLikelihood, state);
        double[] second = operator.jumpAndOptimize(new double[0], start, 7L, 0, logLikelihood, state);

        assertThat(second).containsExactly(first);
    }

    @Test
    public void testHastingsRatioDependsOnlyOnTheNoise() {
        int dimension = 3;
        LargeJumpMALAOperator operator = newOperator(new FakeAdapter(dimension), dimension, 0);
        operator.optimizationCovarianceSampler = trainedSampler(dimension);
        operator.noiseScale = 0.7;

        // the noise terms are the deviations of x and y from a shared mean; the mean itself must
        // not affect the ratio, only those deviations do
        double[] forwardDeviation = {0.4, -0.2, 0.1};
        double[] reverseDeviation = {-0.3, 0.5, -0.15};

        double withFirstMean = ratioForMean(operator, new double[]{0.0, 0.0, 0.0}, forwardDeviation, reverseDeviation);
        double withSecondMean = ratioForMean(operator, new double[]{2.0, -1.0, 3.0}, forwardDeviation, reverseDeviation);

        assertThat(withFirstMean).isCloseTo(withSecondMean, within(1e-9));
    }

    private static double ratioForMean(LargeJumpMALAOperator operator, double[] mean,
                                       double[] forwardDeviation, double[] reverseDeviation) {
        double[] x = add(mean, reverseDeviation);
        double[] y = add(mean, forwardDeviation);
        // logDensity(x; reverseMean=mean) - logDensity(y; forwardMean=mean)
        return operator.logDensity(x, mean) - operator.logDensity(y, mean);
    }

    private static LargeJumpMALAOperator newOperator(FakeAdapter adapter, int dimension, int steps) {
        Randomizer.setSeed(1L);

        LargeJumpMALAOperator operator = new LargeJumpMALAOperator();
        operator.jumpAdapters = new ArrayList<>();
        operator.numJumpMutable = 0;
        operator.optimizationAdapters = List.of(adapter);
        operator.numOptimizationMutable = dimension;
        operator.optimizationOperators = List.of(new RandomWalkOperator(adapter, 0.5));
        operator.numOptimizationSteps = steps;
        operator.jumpScale = 5.0;
        operator.noiseScale = 1.0;
        operator.optimizationCovarianceSampler = new CenteredMultivariateNormalSampler(dimension);
        return operator;
    }

    private static CenteredMultivariateNormalSampler trainedSampler(int dimension) {
        CenteredMultivariateNormalSampler sampler = new CenteredMultivariateNormalSampler(dimension);
        Random random = new Random(3);
        for (int n = 0; n < 200; n++) {
            double[] sample = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                sample[i] = random.nextGaussian();
            }
            sampler.record(new double[] {}, sample);
        }
        return sampler;
    }

    private static double negativeSquaredNorm(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value * value;
        }
        return -sum;
    }

    private static double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    /** Minimal adapter backed by a mutable double[], with identity transforms. */
    private static final class FakeAdapter implements Adapter {
        private final double[] values;

        private FakeAdapter(int dimension) {
            this.values = new double[dimension];
        }

        @Override
        public int getNumImmutable() {
            return 0;
        }

        @Override
        public int getNumMutable() {
            return this.values.length;
        }

        @Override
        public double[] getImmutable(int nodeId) {
            return new double[0];
        }

        @Override
        public double[] getMutable(int nodeId) {
            return this.values.clone();
        }

        @Override
        public double update(double[] mutable, int nodeId) {
            System.arraycopy(mutable, 0, this.values, 0, this.values.length);
            return 0.0;
        }

        @Override
        public double getLogJacobianCorrection(int nodeId) {
            return 0.0;
        }

        @Override
        public List<StateNode> listStateNodes() {
            return List.of();
        }
    }

    /** Fake operator that perturbs the adapter's values with a Gaussian random walk. */
    private static final class RandomWalkOperator extends Operator {
        private final FakeAdapter adapter;
        private final double stepSize;

        private RandomWalkOperator(FakeAdapter adapter, double stepSize) {
            this.adapter = adapter;
            this.stepSize = stepSize;
        }

        @Override
        public void initAndValidate() {
        }

        @Override
        public double proposal() {
            for (int i = 0; i < this.adapter.values.length; i++) {
                this.adapter.values[i] += this.stepSize * Randomizer.nextGaussian();
            }
            return 0.0;
        }

        @Override
        public List<StateNode> listStateNodes() {
            return List.of();
        }
    }

}
