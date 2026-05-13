package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.Int;
import beast.base.spec.inference.parameter.IntVectorParam;

public class IntVectorIdentityTransform extends BEASTObject implements IntVectorTransform<IntVectorParam<? extends Int>> {

    public final Input<IntVectorParam<? extends Int>> parameterInput = new Input<>("parameter", "");

    private IntVectorParam<? extends Int> parameter;

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
        for (int i = 0; i < value.length; i++) {
            this.parameter.set(i, value[i]);
        }
    }

    @Override
    public double getLogJacobianCorrection() {
        return 0;
    }

    @Override
    public IntVectorParam<? extends Int> getStateNode() {
        return this.parameter;
    }

}
