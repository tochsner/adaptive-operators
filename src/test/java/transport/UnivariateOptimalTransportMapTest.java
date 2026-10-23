package transport;

import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Set;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class UnivariateOptimalTransportMapTest {

    private static final Set<Double> pathHeights = Set.of();

    @Test
    void samplesInsideOpenIntervalWithoutEvaluatingEndpoints() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> {
            if (x <= 0.0 || x >= 1.0) {
                throw new AssertionError("density evaluated at endpoint");
            }
            return -x;
        });

        for (int i = 0; i < 20; i++) {
            double sample = map.sample();

            assertThat(sample).isGreaterThan(0.0);
            assertThat(sample).isLessThan(1.0);
        }
    }

    @Test
    void ignoresAdditionalHeightsOutsideSamplingSupport() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, Set.of(0.0, 0.5, 1.0), x -> {
            if (x <= 0.0 || x >= 1.0) {
                throw new AssertionError("density evaluated outside sampling support");
            }
            return -x;
        });

        assertThat(map.logDensity(0.5)).isFinite();
    }

    @Test
    void logDensityRejectsValuesOutsideSamplingSupport() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThatThrownBy(() -> map.logDensity(1.0E-8))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("sampling support");
    }

    @Test
    void supportReachesCloseToTheEndpoints() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThat(map.isInsideSamplingSupport(1.0E-4)).isTrue();
        assertThat(map.isInsideSamplingSupport(1.0 - 1.0E-4)).isTrue();
        assertThat(map.logDensity(1.0E-4)).isFinite();
    }

    @Test
    void logDensityUsesSameShiftedDensityAsSampler() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThat(map.logDensity(0.99) - map.logDensity(0.01)).isCloseTo(-0.98, within(1e-12));
    }

    @Test
    void logHRCorrectionReturnsReverseMinusForwardProposalDensity() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThat(map.logHRCorrection(0.25, 0.75)).isCloseTo(0.5, within(1e-12));
    }

    @Test
    void evaluatesTheDensitySixTimes() {
        int[] evaluations = {0};
        new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> {
            evaluations[0]++;
            return -x;
        });

        assertThat(evaluations[0]).isEqualTo(6);
    }

    @Test
    void excludingSegmentNeverProposesTheCurrentSegment() {
        Randomizer.setSeed(3);
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -3.0 * x);
        double[] boundaries = {0.3, 0.6};

        for (int i = 0; i < 1000; i++) {
            double[] proposal = map.sampleExcludingSegment(0.45, boundaries);

            assertThat(proposal).isNotNull();
            assertThat(proposal[0] < 0.3 || proposal[0] >= 0.6).isTrue();
        }
    }

    @Test
    void excludingSegmentHastingsRatioRenormalisesOverTheOtherSegments() {
        Randomizer.setSeed(4);
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -3.0 * x);
        double[] boundaries = {0.3, 0.6};

        double[] proposal = map.sampleExcludingSegment(0.45, boundaries);
        double currentMass = map.massFractionBelow(0.6) - map.massFractionBelow(0.3);
        double proposedMass = proposal[0] < 0.3
                ? map.massFractionBelow(0.3)
                : 1.0 - map.massFractionBelow(0.6);
        double expected = (map.logDensity(0.45) - Math.log(1.0 - proposedMass))
                - (map.logDensity(proposal[0]) - Math.log(1.0 - currentMass));

        assertThat(proposal[1]).isCloseTo(expected, within(1e-12));
    }

    @Test
    void excludingSegmentFailsWithoutAnotherSegment() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThat(map.sampleExcludingSegment(0.5, new double[0])).isNull();
    }

    @Test
    void excludingSegmentKernelLeavesTheTargetInvariant() {
        // a coarse map of a multimodal target, so a wrong Hastings ratio would bias the histogram
        DoubleUnaryOperator logTarget = x -> Math.log(
                0.6 * normal(x, 2.0, 0.3) + 0.3 * normal(x, 6.0, 0.8) + 0.1 * normal(x, 8.7, 0.15));
        double[] boundaries = {1.5, 3.0, 4.5, 7.0, 8.0, 9.2};
        Set<Double> heights = Arrays.stream(boundaries).boxed().collect(Collectors.toSet());

        Randomizer.setSeed(11);
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 10.0, heights, logTarget::applyAsDouble);
        double lo = 0.0;
        double hi = 10.0;

        int bins = 50;
        int steps = 400_000;
        double[] counts = new double[bins];
        double x = 2.0;
        for (int step = 0; step < steps; step++) {
            double[] proposal = map.sampleExcludingSegment(x, boundaries);
            if (proposal != null) {
                double logAlpha = logTarget.applyAsDouble(proposal[0]) - logTarget.applyAsDouble(x) + proposal[1];
                if (logAlpha >= 0 || Randomizer.nextDouble() < Math.exp(logAlpha)) {
                    x = proposal[0];
                }
            }
            counts[Math.min(bins - 1, (int) ((x - lo) / (hi - lo) * bins))]++;
        }

        // exact bin masses by midpoint integration
        double[] exact = new double[bins];
        double total = 0.0;
        for (int b = 0; b < bins; b++) {
            for (int k = 0; k < 200; k++) {
                exact[b] += Math.exp(logTarget.applyAsDouble(lo + (b + (k + 0.5) / 200) * (hi - lo) / bins));
            }
            total += exact[b];
        }

        double totalVariation = 0.0;
        for (int b = 0; b < bins; b++) {
            totalVariation += 0.5 * Math.abs(counts[b] / steps - exact[b] / total);
        }

        // the correct ratio gives about 0.007 here, omitting the renormalisation gives about 0.09
        assertThat(totalVariation).isLessThan(0.02);
    }

    private static double normal(double x, double mean, double sd) {
        double z = (x - mean) / sd;
        return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
    }

    @Test
    void logHRCorrectionRejectsValuesOutsideSamplingSupport() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThat(map.logHRCorrection(1.0E-8, 0.75)).isEqualTo(Double.NEGATIVE_INFINITY);
    }

}
