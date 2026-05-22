package slice;

import beast.base.inference.Operator;

import java.util.function.Supplier;

public abstract class SliceOperator extends Operator {

    public abstract double proposal(Supplier<Double> computeCurrentLogLikelihood);

    @Override
    public double proposal() {
        throw new UnsupportedOperationException();
    }

}
