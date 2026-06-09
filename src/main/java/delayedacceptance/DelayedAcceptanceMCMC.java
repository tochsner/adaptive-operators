package delayedacceptance;

import beast.base.core.Input;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

public class DelayedAcceptanceMCMC extends MCMC {

    public final Input<PosteriorApproximation> posteriorApproximationInput = new Input<>("approximation", "");
    private PosteriorApproximation posteriorApproximation;
    private boolean reportedUsingApproximation;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.posteriorApproximation = this.posteriorApproximationInput.get();
        this.reportedUsingApproximation = false;
    }

    @Override
    protected Operator propagateState(long sampleNr) {
        if (!this.posteriorApproximation.isReady()) {
            double[] previousValues = this.posteriorApproximation.getCurrentValues();
            double previousLogLikelihood = this.posterior.calculateLogP();

            Operator result = super.propagateState(sampleNr);

            double newLogLikelihood = this.posterior.calculateLogP();

            if (previousLogLikelihood == newLogLikelihood) return result;

            double delta = newLogLikelihood - previousLogLikelihood;
            this.posteriorApproximation.registerLogPosteriorDifference(delta, previousValues);

            return result;
        }

        if (!this.reportedUsingApproximation) {
            System.out.println("Using posterior approximation");
            this.reportedUsingApproximation = true;
        }

        state.store(sampleNr);

        double[] oldValues = this.posteriorApproximation.getCurrentValues();

        final Operator operator = operatorSchedule.selectOperator();
        final double logHastingsRatio = operator.proposal();

        double approximateLogLikelihoodDelta = this.posteriorApproximation.approximateLogPosteriorDifference(oldValues);

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {

            if (operator.requiresStateInitialisation()) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
            }

            double approximateLogAlpha = approximateLogLikelihoodDelta + logHastingsRatio;

            boolean accepted = false;
            if (
                    approximateLogAlpha >= 0 ||
                            (approximateLogAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(approximateLogAlpha))
            ) {
                // we pass the first stage
                // let's look at the real likelihood
                newLogLikelihood = posterior.calculateLogP();
                logAlpha = newLogLikelihood - oldLogLikelihood - approximateLogLikelihoodDelta;

                this.posteriorApproximation.registerLogPosteriorDifference(newLogLikelihood - oldLogLikelihood, oldValues);

                if (logAlpha >= 0 || (logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(logAlpha))) {
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
