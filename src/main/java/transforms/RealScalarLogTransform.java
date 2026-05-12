package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.NonNegativeReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealScalarParam;

public class RealScalarLogTransform extends BEASTObject implements RealScalarTransform<RealScalarParam<? extends Real>> {

    public final Input<? extends RealScalarParam<? extends NonNegativeReal>> parameterInput = new Input<>("parameter", "");

    private RealScalarParam<? extends NonNegativeReal> parameter;

    @Override
    public void initAndValidate() {
        this.parameter = this.parameterInput.get();
    }

    @Override
    public Double get() {
        return Math.log(this.parameter.get());
    }

    @Override
    public void set(Double value) {
        this.parameter.set(Math.exp(value));
    }

    @Override
    public double getLogJacobianCorrection() {
        return -Math.log(this.parameter.get());
    }

    @Override
    public RealScalarParam<? extends Real> getStateNode() {
        return this.parameter;
    }

}
