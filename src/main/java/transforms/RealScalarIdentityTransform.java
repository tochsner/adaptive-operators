package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealScalarParam;

public class RealScalarIdentityTransform extends BEASTObject implements RealScalarTransform<RealScalarParam<? extends Real>> {

    public final Input<RealScalarParam<? extends Real>> parameterInput = new Input<>("parameter", "");

    private RealScalarParam<? extends Real> parameter;

    @Override
    public void initAndValidate() {
        this.parameter = this.parameterInput.get();
    }

    @Override
    public Double get() {
        return this.parameter.get();
    }

    @Override
    public void set(Double value) {
        this.parameter.set(value);
    }

    @Override
    public double getLogJacobianCorrection() {
        return 0;
    }

    @Override
    public RealScalarParam<? extends Real> getStateNode() {
        return this.parameter;
    }

}
