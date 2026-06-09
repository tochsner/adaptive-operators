package delayedacceptance;

import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

public class ApproximateDelayedAcceptanceMCMC extends MCMC {

    @Override
    public void initAndValidate() {
        super.initAndValidate();
    }

    @Override
    protected Operator propagateState(long sampleNr) {
        if (!(this.posterior instanceof ApproximateDistribution approximatePosterior)) return super.propagateState(sampleNr);

        state.store(sampleNr);

        double oldApproximateLogLikelihood = approximatePosterior.calculateApproximateLogP();

        final Operator operator = operatorSchedule.selectOperator();
        final double logHastingsRatio = operator.proposal();

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {

            if (operator.requiresStateInitialisation()) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
            }

            double newApproximateLogLikelihood = approximatePosterior.calculateApproximateLogP();

            double approximateLogAlpha = newApproximateLogLikelihood - oldApproximateLogLikelihood + logHastingsRatio;

            boolean accepted = false;
            boolean evaluatedFullPosterior = false;
            if (
                    approximateLogAlpha >= 0 ||
                            (approximateLogAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(approximateLogAlpha))
            ) {
                // we pass the first stage
                // let's look at the real likelihood
                newLogLikelihood = posterior.calculateLogP();
                evaluatedFullPosterior = true;

                double fullLogAlpha = newLogLikelihood - oldLogLikelihood + logHastingsRatio;
                double correctionLogAlpha = fullLogAlpha - approximateLogAlpha;
                logAlpha = fullLogAlpha;

                if (correctionLogAlpha >= 0 || (correctionLogAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(correctionLogAlpha))) {
                    // accept
                    oldLogLikelihood = newLogLikelihood;
                    state.acceptCalculationNodes();

                    if (sampleNr >= 0) {
                        operator.accept();
                    }
                    accepted = true;
                }
            }

            if (!accepted) {
                // reject
                if (!evaluatedFullPosterior) {
                    logAlpha = Double.NEGATIVE_INFINITY;
                }
                if (sampleNr >= 0) {
                    operator.reject(0);
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
