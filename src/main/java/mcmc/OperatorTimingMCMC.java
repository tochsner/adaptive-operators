package mcmc;

import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import weightoptimization.ESSWeightAdaptationOperator;

/**
 * MCMC driver that measures how long each step takes, so that operators can be compared by their
 * cost per application.
 *
 * <p>A step is timed as a whole: the proposal, the recomputation of the posterior and the accept or
 * reject. The cost of an operator therefore includes how much of the likelihood it lets BEAST cache,
 * not just the time its own proposal takes. The duration is handed to every
 * {@link ESSWeightAdaptationOperator} that was selected.
 */
public class OperatorTimingMCMC extends JumpMCMC {

    @Override
    protected Operator propagateState(final long sampleNr) {
        long start = System.nanoTime();

        Operator operator = super.propagateState(sampleNr);

        if (operator instanceof ESSWeightAdaptationOperator weightOperator) {
            weightOperator.recordApplicationTime(System.nanoTime() - start);
        }

        return operator;
    }
}
