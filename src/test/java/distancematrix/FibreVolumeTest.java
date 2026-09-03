package distancematrix;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Checks the closed-form fibre volume that {@link ReferenceDensity#logFibreVolume} claims, since the
 * whole correctness argument for {@link DistanceMatrixOperator} rests on it.
 *
 * <p>The claim is that if the matrix entries are drawn independently from the reference density
 * {@code g}, then the tree produced by single linkage has density
 *
 * <pre>
 *     V(T) = prod_v  m_v * g(w_v) * Gbar(w_v)^(m_v - 1)
 * </pre>
 *
 * over ranked labelled topologies and ordered node distances {@code w_v = 2 * h_v}. If that is
 * right, then for any probability density {@code q} on the same space,
 * {@code E_{D ~ g}[ q(SL(D)) / V(SL(D)) ] = 1}. That identity is what these tests measure, using an
 * exponential {@code q} over the ordered node distances and a uniform choice of ranked topology.
 */
class FibreVolumeTest {

    private static final double ENTRY_RATE = 1.0;

    // concentrating q where V has its mass keeps the importance sampling ratio well behaved
    private static final double HEIGHT_RATE = 5.0;

    private static final int SAMPLE_COUNT = 150_000;

    @ParameterizedTest
    @ValueSource(ints = {3, 4, 5, 6})
    void fibreVolumeIntegratesToOne(int leafCount) {
        Random random = new Random(20240830L + leafCount);
        ReferenceDensity density = new ExponentialReferenceDensity(ENTRY_RATE);

        double sum = 0.0;
        double sumOfSquares = 0.0;

        for (int sample = 0; sample < SAMPLE_COUNT; sample++) {
            SingleLinkageTree clustering =
                    new SingleLinkageTree(leafCount, sampleMatrix(leafCount, random));

            double ratio = Math.exp(
                    logReferenceTreeDensity(clustering) - density.logFibreVolume(clustering));

            sum += ratio;
            sumOfSquares += ratio * ratio;
        }

        double estimate = sum / SAMPLE_COUNT;
        double standardError =
                Math.sqrt((sumOfSquares / SAMPLE_COUNT - estimate * estimate) / SAMPLE_COUNT);

        assertThat(standardError).as("estimator is too noisy to be conclusive").isLessThan(0.01);
        assertThat(estimate).as("leafCount = " + leafCount).isCloseTo(1.0, within(0.03));
    }

    @Test
    void crossPairCountsDriveTheCorrection() {
        // the two leaves joined first always form a group of one pair, so they contribute no
        // survival term, while every later merge does
        SingleLinkageTree clustering = new SingleLinkageTree(3, new double[]{1.0, 3.0, 3.0});
        ReferenceDensity density = new ExponentialReferenceDensity(2.0);

        double expected = Math.log(1) + density.logDensity(1.0)
                + Math.log(2) + density.logDensity(3.0) + density.logSurvival(3.0);

        assertThat(density.logFibreVolume(clustering)).isCloseTo(expected, within(1e-12));
    }

    @Test
    void exponentialDensityAndSurvivalAgree() {
        ReferenceDensity density = new ExponentialReferenceDensity(3.0);

        assertThat(density.logDensity(0.7)).isCloseTo(Math.log(3.0) - 3.0 * 0.7, within(1e-12));
        assertThat(density.logSurvival(0.7)).isCloseTo(-3.0 * 0.7, within(1e-12));
        assertThat(density.logDensity(0.0)).isNegative().isInfinite();
        assertThat(density.logSurvival(-1.0)).isZero();
    }

    @Test
    void logNormalDensityAndSurvivalAgree() {
        ReferenceDensity density = new LogNormalReferenceDensity(0.0, 1.0);

        // the median of a standard log-normal is one, so half the mass lies above it
        assertThat(density.logSurvival(1.0)).isCloseTo(Math.log(0.5), within(1e-9));
        assertThat(density.logDensity(1.0))
                .isCloseTo(-Math.log(Math.sqrt(2.0 * Math.PI)), within(1e-9));
        assertThat(density.logDensity(0.0)).isNegative().isInfinite();
    }

    private static double[] sampleMatrix(int leafCount, Random random) {
        double[] distances = new double[leafCount * (leafCount - 1) / 2];

        for (int pair = 0; pair < distances.length; pair++) {
            distances[pair] = -Math.log(random.nextDouble()) / ENTRY_RATE;
        }

        return distances;
    }

    /**
     * A reference probability density over the same space the fibre volume lives on: a uniform
     * choice among the labelled ranked topologies, times the density of the ordered node distances
     * under independent exponentials.
     */
    private static double logReferenceTreeDensity(SingleLinkageTree clustering) {
        int leafCount = clustering.getLeafCount();

        double sumOfDistances = 0.0;
        for (int merge = 0; merge < leafCount - 1; merge++) {
            sumOfDistances += 2.0 * clustering.getMergeHeight(merge);
        }

        return logFactorial(leafCount - 1)
                - logRankedTopologyCount(leafCount)
                + (leafCount - 1) * Math.log(HEIGHT_RATE)
                - HEIGHT_RATE * sumOfDistances;
    }

    /** Number of labelled ranked topologies on {@code n} taxa, which is n! (n-1)! / 2^(n-1). */
    private static double logRankedTopologyCount(int leafCount) {
        return logFactorial(leafCount) + logFactorial(leafCount - 1)
                - (leafCount - 1) * Math.log(2.0);
    }

    private static double logFactorial(int value) {
        double result = 0.0;
        for (int factor = 2; factor <= value; factor++) {
            result += Math.log(factor);
        }
        return result;
    }
}
