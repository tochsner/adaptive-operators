package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.inference.State;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Checks that the likelihood cache stays exact when the operator clears node dirty flags between
 * map evaluations and marks only the nodes it touched after a rejection. Every cached value is
 * compared with an independent copy of the model that recalculates from scratch.
 */
class GuidedNodeTransportOperatorCacheTest {

    @Test
    void cachedLikelihoodsMatchAFullRecalculation() {
        ChainStatistics statistics = runChain(new GuidedNodeTransportOperator(), 400);

        // both the accept and the recovery path have to be exercised
        assertThat(statistics.accepted()).isPositive();
        assertThat(statistics.recovered()).isPositive();
    }

    @Test
    void comparisonDetectsAStaleCache() {
        // a recovery that marks nothing leaves the cache at the last evaluated tree
        GuidedNodeTransportOperator marksNothing = new GuidedNodeTransportOperator() {
            @Override
            public void markDirtyAfterRestore(State state) {
            }
        };

        assertThatThrownBy(() -> runChain(marksNothing, 400)).isInstanceOf(AssertionError.class);
    }

    private record ChainStatistics(int accepted, int recovered) {}

    /** Runs the operator the way JumpMCMC does and checks every likelihood the cache produces. */
    private static ChainStatistics runChain(GuidedNodeTransportOperator operator, int proposals) {
        Alignment data = SmallModel.createAlignment();
        SmallModel model = SmallModel.create(data);
        SmallModel reference = SmallModel.create(data);

        operator.initByName("tree", model.tree(), "alignment", data, "weightingScheme", "uniform", "weight", 1.0);

        Supplier<Double> computeCurrentLogLikelihood = () -> {
            model.state().checkCalculationNodesDirtiness();
            return assertExact(model, reference, model.posterior().calculateLogP());
        };

        Randomizer.setSeed(5);
        double currentLogP = model.state().robustlyCalcPosterior(model.posterior());
        int accepted = 0;
        int recovered = 0;

        for (int i = 0; i < proposals; i++) {
            model.state().store(i);
            double logHR = operator.proposal(computeCurrentLogLikelihood, model.state());

            if (logHR != Double.NEGATIVE_INFINITY) {
                model.state().checkCalculationNodesDirtiness();
                double proposedLogP = assertExact(model, reference, model.posterior().calculateLogP());
                double logAlpha = proposedLogP - currentLogP + logHR;

                if (logAlpha >= 0 || Randomizer.nextDouble() < Math.exp(logAlpha)) {
                    currentLogP = proposedLogP;
                    model.state().acceptCalculationNodes();
                    accepted++;
                    model.state().setEverythingDirty(false);
                    continue;
                }
            }

            // rejected or failed: the recovery JumpMCMC performs after state.restore()
            double recoveredLogP = assertExact(model, reference, model.recoverAfterRejection(operator));
            recovered++;

            assertThat(recoveredLogP).isCloseTo(currentLogP, within(1e-9));
        }

        return new ChainStatistics(accepted, recovered);
    }

    private static double assertExact(SmallModel model, SmallModel reference, double cachedLogP) {
        reference.tree().assignFromFragile(model.tree());
        double exactLogP = reference.state().robustlyCalcPosterior(reference.posterior());
        assertThat(cachedLogP).isCloseTo(exactLogP, within(1e-9));
        return cachedLogP;
    }
}
