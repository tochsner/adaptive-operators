package surrogate;

import beast.base.core.Input;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

public class SimulatedAnnealing extends MCMC {

    final public Input<Double> temperatureInput =
            new Input<>("temperature", "temperature that scales the acceptance criterion; " +
                    "higher values accept downhill moves more readily, values approaching 0 " +
                    "reduce to greedy hill climbing", 0.5);

    double temperature;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.temperature = this.temperatureInput.get();
    }

    @Override
    protected Operator propagateState(long sampleNr) {
        state.store(sampleNr);

        final Operator operator = operatorSchedule.selectOperator();
        final double logHastingsRatio = operator.proposal();

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {

            if (operator.requiresStateInitialisation()) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
            }

            newLogLikelihood = posterior.calculateLogP();

            // simulated annealing acceptance: scale the log-likelihood difference by the
            // temperature and keep the Hastings ratio as an unscaled proposal correction.
            // a worse move is then accepted with probability exp(logAlpha), so the penalty
            // grows with how much worse the move is rather than being a flat chance
            logAlpha = (newLogLikelihood - oldLogLikelihood) / this.temperature + logHastingsRatio;

            final boolean accept = logAlpha >= 0
                    || (logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(logAlpha));

            if (accept) {
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
