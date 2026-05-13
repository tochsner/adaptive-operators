package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealVectorParam;

public class RealVectorIdentityTransform extends BEASTObject implements RealVectorTransform<RealVectorParam<? extends Real>> {

    public final Input<RealVectorParam<? extends Real>> parameterInput = new Input<>("parameter", "");

    private RealVectorParam<? extends Real> parameter;

    @Override
    public void initAndValidate() {
        this.parameter = this.parameterInput.get();
    }

    @Override
    public Double[] get() {
        return this.parameter.getElements().toArray(new Double[0]);
    }

    @Override
    public void set(Double[] value) {
        for (int i = 0; i < value.length; i++) {
            this.parameter.set(i, value[i]);
        }
    }

    @Override
    public double getLogJacobianCorrection() {
        return 0;
    }

    @Override
    public double getLogJacobianCorrection(int index) {
        return 0;
    }

    @Override
    public RealVectorParam<? extends Real> getStateNode() {
        return this.parameter;
    }

}
