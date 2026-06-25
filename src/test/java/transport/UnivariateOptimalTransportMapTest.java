package transport;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class UnivariateOptimalTransportMapTest {

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
    void logHRCorrectionReturnsReverseMinusForwardProposalDensity() {
        UnivariateOptimalTransportMap map = new UnivariateOptimalTransportMap(0, 1.0, pathHeights, x -> -x);

        assertThat(map.logHRCorrection(0.25, 0.75)).isCloseTo(0.5, within(1e-12));
    }

}
