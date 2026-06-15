package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.NonNegativeInt;
import beast.base.spec.inference.parameter.IntSimplexParam;

public class IntSimplexRoundTransform extends BEASTObject implements IntVectorTransform<IntSimplexParam<? extends NonNegativeInt>> {

    public final Input<IntSimplexParam<? extends NonNegativeInt>> parameterInput = new Input<>("parameter", "");

    private IntSimplexParam<? extends NonNegativeInt> parameter;

    @Override
    public void initAndValidate() {
        this.parameter = this.parameterInput.get();
    }

    @Override
    public Integer[] get() {
        return this.parameter.getElements().toArray(new Integer[0]);
    }

    @Override
    public void set(Integer[] value) {
        int sum = 0;

        for (Integer element : value) {
            sum += element;
        }

        if (sum != this.parameter.expectedSum()) {
            throw new IllegalArgumentException("Expected integer simplex values to sum to "
                    + this.parameter.expectedSum() + ", but got " + sum);
        }

        for (int i = 0; i < value.length; i++) {
            this.parameter.set(i, value[i]);
        }
    }

    @Override
    public double getLogJacobianCorrection() {
        return 0.0;
    }

    @Override
    public IntSimplexParam<? extends NonNegativeInt> getStateNode() {
        return this.parameter;
    }

}
