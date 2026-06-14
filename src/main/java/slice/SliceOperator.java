package slice;

import beast.base.inference.Operator;
import beast.base.inference.State;

import java.util.function.Supplier;

public abstract class SliceOperator extends Operator {

    public abstract double proposal(Supplier<Double> computeCurrentLogLikelihood, State state);

    @Override
    public double proposal() {
        throw new UnsupportedOperationException();
    }

}
