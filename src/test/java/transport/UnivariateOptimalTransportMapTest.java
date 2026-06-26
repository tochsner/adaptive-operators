package transport;

import org.junit.jupiter.api.Test;

import java.util.Set;

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

        assertThatThrownBy(() -> map.logDensity(0.005))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("sampling support");
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
    void logHRCorrectionRejectsValuesOutsideSamplingSupport() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThat(map.logHRCorrection(0.005, 0.75)).isEqualTo(Double.NEGATIVE_INFINITY);
    }

}
