package mcmc;

import beast.base.inference.*;
import beast.base.util.Randomizer;
import slice.SliceOperator;

import java.util.function.Supplier;

public class SliceMCMC extends MCMC {

    protected Operator propagateState(final long sampleNr) {
        state.store(sampleNr);

        final Operator operator = operatorSchedule.selectOperator();

        Supplier<Double> computeCurrentLogLikelihood = () -> {
            if (operator.requiresStateInitialisation()) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
            }

            return posterior.calculateLogP();
        };

        final double logHastingsRatio;
        if (operator instanceof SliceOperator sliceOperator) {
            logHastingsRatio = sliceOperator.proposal(computeCurrentLogLikelihood);
        } else {
            logHastingsRatio = operator.proposal();
        }

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {

            if (operator.requiresStateInitialisation()) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
            }

            newLogLikelihood = posterior.calculateLogP();
            logAlpha = newLogLikelihood - oldLogLikelihood + logHastingsRatio; //CHECK HASTINGS

            if (logAlpha >= 0 || (logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(logAlpha))) {
                // accept
                oldLogLikelihood = newLogLikelihood;
                state.acceptCalculationNodes();

                if (sampleNr >= 0) {
                    operator.accept();
                }
            } else {
                // reject
                if (sampleNr >= 0) {
                    operator.reject(newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                }
                state.restore();
                state.restoreCalculationNodes();
            }
            state.setEverythingDirty(false);
        } else {
            logAlpha = Double.NEGATIVE_INFINITY;
            // operation failed
            if (sampleNr >= 0) {
                operator.reject(-2);
            }
            state.restore();
            if (!operator.requiresStateInitialisation()) {
                state.setEverythingDirty(false);
                state.restoreCalculationNodes();
            }
        }
        log(sampleNr);
        return operator;
    }

}
