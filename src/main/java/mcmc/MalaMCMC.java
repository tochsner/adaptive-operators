package mcmc;

import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import mala.FisherMALAOperator;
import mala.MALAOperator;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class MalaMCMC extends MCMC {

    private Map<Operator, Function<Double, Void>> recordCallbacks = new HashMap<>();

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        for (Operator operator : this.operatorSchedule.operators) {
            if (operator instanceof MALAOperator malaOperator) {
                this.recordCallbacks.putIfAbsent(operator, malaOperator.getRecordCallback());
            } else if (operator instanceof FisherMALAOperator fisherMALAOperator) {
                this.recordCallbacks.putIfAbsent(operator, fisherMALAOperator.getRecordCallback());
            }
        }
    }

    protected Operator propagateState(final long sampleNr) {
        if (Randomizer.nextDouble() < 0.05) {
            for (Function<Double, Void> recordCallback : this.recordCallbacks.values()) {
                recordCallback.apply(oldLogLikelihood);
            }
        }
        return super.propagateState(sampleNr);
    }

}
