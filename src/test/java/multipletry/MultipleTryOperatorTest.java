package multipletry;

import adapters.Adapter;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import multipletry.MultipleTryOperator.BalancingFunction;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class MultipleTryOperatorTest {

    @ParameterizedTest
    @EnumSource(BalancingFunction.class)
    public void testTargetsAnAnisotropicGaussian(BalancingFunction balancingFunction) {
        Randomizer.setSeed(1L);

        FakeAdapter adapter = new FakeAdapter(2);
        MultipleTryOperator operator = newOperator(adapter, 10);
        operator.balancingFunction = balancingFunction;

        // independent normals with standard deviations one and three
        Supplier<Double> logP = () -> -0.5 * (Math.pow(adapter.values[0], 2.0)
                + Math.pow(adapter.values[1], 2.0) / 9.0);

        double[][] samples = runChain(operator, adapter, logP, 50_000);

        assertThat(mean(samples, 0)).isCloseTo(0.0, within(0.15));
        assertThat(mean(samples, 1)).isCloseTo(0.0, within(0.45));

        // the reverse neighbourhood is what keeps these honest: without it the chain is pulled
        // towards the high-density candidates and the spread comes out too small
        assertThat(standardDeviation(samples, 0)).isCloseTo(1.0, within(0.1));
        assertThat(standardDeviation(samples, 1)).isCloseTo(3.0, within(0.3));
    }

    @Test
    public void testAccountsForTheAdapterLogJacobian() {
        Randomizer.setSeed(2L);

        // the mutable coordinate is u = log x, so du/dx = exp(-u) and the correction is -u
        FakeAdapter adapter = new FakeAdapter(1);
        adapter.logJacobianCorrection = values -> -values[0];

        MultipleTryOperator operator = newOperator(adapter, 10);

        // an Exponential(1) target on x, stated in x as the posterior would be
        Supplier<Double> logP = () -> -Math.exp(adapter.values[0]);

        double[][] samples = runChain(operator, adapter, logP, 50_000);

        double[] x = new double[samples.length];
        for (int i = 0; i < samples.length; i++) {
            x[i] = Math.exp(samples[i][0]);
        }

        // drop the Jacobian and the target in u becomes exp(-exp(u)), whose image in x is the
        // improper exp(-x)/x, so the chain drifts towards zero instead of settling here
        assertThat(average(x)).isCloseTo(1.0, within(0.15));
        assertThat(spread(x)).isCloseTo(1.0, within(0.2));
    }

    @ParameterizedTest
    @EnumSource(BalancingFunction.class)
    public void testEveryBalancingFunctionSatisfiesTheBalancingCondition(BalancingFunction balancingFunction) {
        MultipleTryOperator operator = newOperator(new FakeAdapter(1), 1);
        operator.balancingFunction = balancingFunction;

        // g(t) = t * g(1/t) is exactly what makes a weight admissible: it is the locally balanced
        // condition, and it is also what the symmetry of the Liu-Liang-Wong lambda reduces to. In
        // logs it says the two weights of a pair differ by the log density ratio, which is also
        // why a single try collapses to plain Metropolis.
        double[][] pairs = {{-3.0, -5.0}, {0.0, 0.0}, {12.0, 4.0}, {-1.5, 100.0}, {7.25, -7.25}};

        for (double[] pair : pairs) {
            double difference = operator.logBalancingWeight(pair[0], pair[1])
                    - operator.logBalancingWeight(pair[1], pair[0]);

            assertThat(difference).isCloseTo(pair[0] - pair[1], within(1.0E-9));
        }

        // an impossible candidate is never selected, whichever function is in use
        assertThat(operator.logBalancingWeight(Double.NEGATIVE_INFINITY, 0.0))
                .isEqualTo(Double.NEGATIVE_INFINITY);

        // and an impossible reference leaves every reachable candidate equally weighted rather
        // than producing an infinity that would make the acceptance ratio a NaN
        assertThat(operator.logBalancingWeight(3.0, Double.NEGATIVE_INFINITY)).isEqualTo(0.0);
    }

    @Test
    public void testBarkerWeighsEqualDensitiesAtOneHalf() {
        MultipleTryOperator operator = newOperator(new FakeAdapter(1), 1);
        operator.balancingFunction = BalancingFunction.BARKER;

        assertThat(operator.logBalancingWeight(7.0, 7.0)).isCloseTo(-Math.log(2.0), within(1.0E-12));
    }

    @Test
    public void testSingleTryTargetsAStandardNormal() {
        Randomizer.setSeed(3L);

        FakeAdapter adapter = new FakeAdapter(1);
        MultipleTryOperator operator = newOperator(adapter, 1);

        Supplier<Double> logP = () -> -0.5 * Math.pow(adapter.values[0], 2.0);

        double[][] samples = runChain(operator, adapter, logP, 50_000);

        assertThat(mean(samples, 0)).isCloseTo(0.0, within(0.15));
        assertThat(standardDeviation(samples, 0)).isCloseTo(1.0, within(0.1));
    }

    @Test
    public void testGlobalScaleIsPulledBackTowardsItsEquilibrium() {
        Randomizer.setSeed(4L);

        // the per-coordinate running standard deviations already absorb the scale of the target,
        // so the global scale is effectively unitless and settles wherever the target acceptance
        // rate puts it; what matters is that it is pulled there from either side

        FakeAdapter bold = new FakeAdapter(1);
        MultipleTryOperator boldOperator = newOperator(bold, 5);
        boldOperator.globalScale = 1.0E4;
        runChain(boldOperator, bold, standardNormal(bold), 2_000);

        assertThat(boldOperator.globalScale).isLessThan(1.0E4);

        FakeAdapter timid = new FakeAdapter(1);
        MultipleTryOperator timidOperator = newOperator(timid, 5);
        timidOperator.globalScale = 1.0E-6;
        runChain(timidOperator, timid, standardNormal(timid), 2_000);

        assertThat(timidOperator.globalScale).isGreaterThan(1.0E-6);
    }

    @Test
    public void testGlobalScaleGrowsWhenEverythingIsAccepted() {
        Randomizer.setSeed(5L);

        FakeAdapter adapter = new FakeAdapter(1);
        MultipleTryOperator operator = newOperator(adapter, 5);
        double initialScale = operator.globalScale;

        // a flat target accepts every move, which is well above the target acceptance rate
        Supplier<Double> logP = () -> 0.0;

        runChain(operator, adapter, logP, 2_000);

        assertThat(operator.globalScale).isGreaterThan(initialScale);
    }

    @Test
    public void testLearnsThePerCoordinateStandardDeviations() {
        Randomizer.setSeed(6L);

        FakeAdapter adapter = new FakeAdapter(2);
        MultipleTryOperator operator = newOperator(adapter, 10);

        Supplier<Double> logP = () -> -0.5 * (Math.pow(adapter.values[0], 2.0)
                + Math.pow(adapter.values[1], 2.0) / 9.0);

        runChain(operator, adapter, logP, 50_000);

        double[] standardDeviations = learnedScales(operator, 2);

        assertThat(standardDeviations[0]).isCloseTo(1.0, within(0.1));
        assertThat(standardDeviations[1]).isCloseTo(3.0, within(0.3));
    }

    @Test
    public void testLearnsAScalePerCoordinateKeyWhenCoordinatesAreReshuffled() {
        Randomizer.setSeed(7L);

        // three quantities whose spreads span three orders of magnitude, sitting at a different
        // coordinate every proposal, exactly as the pairs of a redrawn cube do
        double[] targetSpreads = {0.05, 1.0, 20.0};

        ShufflingAdapter adapter = new ShufflingAdapter(targetSpreads.length, new Random(11));
        MultipleTryOperator operator = newOperator(adapter, 10);

        Supplier<Double> logP = () -> {
            double logDensity = 0.0;
            for (int key = 0; key < targetSpreads.length; key++) {
                logDensity -= 0.5 * Math.pow(adapter.values[key] / targetSpreads[key], 2.0);
            }
            return logDensity;
        };

        double[][] samples = runChain(operator, adapter, logP, 50_000);

        // the samples are recorded in coordinate order, so undo the shuffle to read them by key
        for (int key = 0; key < targetSpreads.length; key++) {
            double spread = standardDeviation(samples, key);

            // keying on position would blend all three into one middling number and the chain
            // would fail to explore the widest of them
            assertThat(spread).isCloseTo(targetSpreads[key], within(0.2 * targetSpreads[key]));
        }

        // and the learned scales themselves have to be per key, not one shared value
        double[] learned = learnedScales(operator, targetSpreads.length);

        for (int key = 0; key < targetSpreads.length; key++) {
            assertThat(learned[key]).isCloseTo(targetSpreads[key], within(0.2 * targetSpreads[key]));
        }
    }

    private static Supplier<Double> standardNormal(FakeAdapter adapter) {
        return () -> -0.5 * Math.pow(adapter.values[0], 2.0);
    }

    @Test
    public void testTheScaleLadderPreservesTheTarget() {
        Randomizer.setSeed(8L);

        FakeAdapter adapter = new FakeAdapter(2);
        MultipleTryOperator operator = newOperator(adapter, 10);
        operator.numScales = 5;
        operator.scaleSpacing = 4.0;
        operator.initAdaptation();

        Supplier<Double> logP = () -> -0.5 * (Math.pow(adapter.values[0], 2.0)
                + Math.pow(adapter.values[1], 2.0) / 9.0);

        double[][] samples = runChain(operator, adapter, logP, 50_000);

        // a mixture of symmetric kernels is still symmetric, so spreading the candidates over a
        // ladder must leave the invariant distribution exactly where it was
        assertThat(mean(samples, 0)).isCloseTo(0.0, within(0.15));
        assertThat(standardDeviation(samples, 0)).isCloseTo(1.0, within(0.1));
        assertThat(standardDeviation(samples, 1)).isCloseTo(3.0, within(0.3));
    }

    @Test
    public void testTheScaleLadderCrossesBetweenSeparatedModes() {
        Randomizer.setSeed(9L);

        // far enough apart that a single-scale kernel is genuinely trapped: with the within-mode
        // spread at one, the adapted step is a couple of units against a sixty unit gap. Measured
        // over 50k iterations, one scale first crosses at iteration 2073 and crosses 152 times,
        // while this ladder first crosses at 34 and crosses 1462 times.
        double separation = 30.0;

        FakeAdapter adapter = new FakeAdapter(1);
        MultipleTryOperator operator = newOperator(adapter, 10);
        // three rungs at spacing eight, i.e. {1/8, 1, 8}: the point is to exercise a coarse rung
        // that actually reaches the far mode
        operator.numScales = 3;
        operator.scaleSpacing = 8.0;
        operator.initAdaptation();

        // two unit-spread modes, with the chain started in the left one
        adapter.values[0] = -separation;

        Supplier<Double> logP = () -> {
            double x = adapter.values[0];
            return logAddExp(-0.5 * Math.pow(x + separation, 2.0), -0.5 * Math.pow(x - separation, 2.0));
        };

        double[][] samples = runChain(operator, adapter, logP, 50_000);

        int right = 0;
        int crossings = 0;
        for (int i = 0; i < samples.length; i++) {
            if (samples[i][0] > 0.0) right++;
            if (i > 0 && (samples[i][0] > 0.0) != (samples[i - 1][0] > 0.0)) crossings++;
        }

        assertThat(crossings).isGreaterThan(400);
        assertThat((double) right / samples.length).isCloseTo(0.5, within(0.15));
    }

    private static double logAddExp(double a, double b) {
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    /** The scales the operator has learned for the keys zero to {@code dimension}, exclusive. */
    private static double[] learnedScales(MultipleTryOperator operator, int dimension) {
        long[] keys = new long[dimension];

        for (int i = 0; i < keys.length; i++) {
            keys[i] = i;
        }

        return operator.variances.getStandardDeviations(keys, operator.coordinateAdapterIndices);
    }

    private static MultipleTryOperator newOperator(ArrayBackedAdapter adapter, int numTries) {
        MultipleTryOperator operator = new MultipleTryOperator();
        operator.adapters = List.of(adapter);
        operator.tree = null;
        operator.totalNumMutable = adapter.getNumMutable();
        operator.numTries = numTries;
        operator.targetAcceptance = 0.44;
        operator.numScales = 1;
        operator.scaleSpacing = 8.0;
        operator.initAdaptation();
        return operator;
    }

    /**
     * Drives the operator the way {@code SliceMCMC.propagateState} does: the operator leaves the
     * state on its proposal and signals a rejection with a negatively infinite Hastings ratio,
     * upon which the driver calls {@code State.restore()}. Restoring the snapshot by hand here is
     * exactly that, so the chain this builds is the real kernel.
     */
    private static double[][] runChain(MultipleTryOperator operator, ArrayBackedAdapter adapter,
                                       Supplier<Double> logP, int iterations) {
        double[][] samples = new double[iterations][];

        for (int i = 0; i < iterations; i++) {
            double[] before = adapter.values().clone();

            if (operator.proposal(logP, null) == Double.NEGATIVE_INFINITY) {
                System.arraycopy(before, 0, adapter.values(), 0, before.length);
            }

            samples[i] = adapter.values().clone();
        }

        return samples;
    }

    private static double mean(double[][] samples, int coordinate) {
        double sum = 0.0;
        for (double[] sample : samples) {
            sum += sample[coordinate];
        }
        return sum / samples.length;
    }

    private static double standardDeviation(double[][] samples, int coordinate) {
        double mean = mean(samples, coordinate);
        double sum = 0.0;
        for (double[] sample : samples) {
            sum += Math.pow(sample[coordinate] - mean, 2.0);
        }
        return Math.sqrt(sum / (samples.length - 1));
    }

    private static double average(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static double spread(double[] values) {
        double mean = average(values);
        double sum = 0.0;
        for (double value : values) {
            sum += Math.pow(value - mean, 2.0);
        }
        return Math.sqrt(sum / (values.length - 1));
    }

    /** An adapter whose state is a plain double[], so the test harness can snapshot it. */
    private interface ArrayBackedAdapter extends Adapter {
        double[] values();
    }

    /** Minimal adapter backed by a mutable double[], with a configurable Jacobian. */
    private static final class FakeAdapter implements ArrayBackedAdapter {

        private final double[] values;
        private ToDoubleFunction<double[]> logJacobianCorrection = values -> 0.0;

        private FakeAdapter(int dimension) {
            this.values = new double[dimension];
        }

        @Override
        public double[] values() {
            return this.values;
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
            return this.logJacobianCorrection.applyAsDouble(this.values);
        }

        @Override
        public List<StateNode> listStateNodes() {
            return List.of();
        }
    }


    /**
     * An adapter that holds its values by identity but exposes them at a position that changes on
     * every refresh, the way {@code CubeAdapter} does when it redraws its cube.
     */
    private static final class ShufflingAdapter implements ArrayBackedAdapter {

        private final double[] values;
        private final int[] positionToKey;
        private final Random random;

        private ShufflingAdapter(int dimension, Random random) {
            this.values = new double[dimension];
            this.positionToKey = new int[dimension];
            this.random = random;

            for (int i = 0; i < dimension; i++) {
                this.positionToKey[i] = i;
            }
        }

        @Override
        public double[] values() {
            return this.values;
        }

        @Override
        public void refresh() {
            for (int i = this.positionToKey.length - 1; i > 0; i--) {
                int j = this.random.nextInt(i + 1);
                int swap = this.positionToKey[i];
                this.positionToKey[i] = this.positionToKey[j];
                this.positionToKey[j] = swap;
            }
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
            double[] mutable = new double[this.values.length];

            for (int i = 0; i < mutable.length; i++) {
                mutable[i] = this.values[this.positionToKey[i]];
            }

            return mutable;
        }

        @Override
        public double update(double[] mutable, int nodeId) {
            for (int i = 0; i < mutable.length; i++) {
                this.values[this.positionToKey[i]] = mutable[i];
            }

            return 0.0;
        }

        @Override
        public int[] getMutableKeys(int nodeId) {
            return this.positionToKey.clone();
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

}
