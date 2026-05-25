package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.UnitInterval;
import beast.base.spec.inference.parameter.RealScalarParam;

public class RealScalarSigmoidTransform extends BEASTObject implements RealScalarTransform<RealScalarParam<? extends UnitInterval>> {

    public final Input<? extends RealScalarParam<? extends UnitInterval>> parameterInput = new Input<>("parameter", "");

    private RealScalarParam<? extends UnitInterval> parameter;

    @Override
    public void initAndValidate() {
        this.parameter = this.parameterInput.get();
    }

    @Override
    public Double get() {
        double value = this.parameter.get();
        return Math.log(value) - Math.log1p(-value);
    }

    @Override
    public void set(Double value) {
        this.parameter.set(1.0 / (1.0 + Math.exp(-value)));
    }

    @Override
    public double getLogJacobianCorrection() {
        double value = this.parameter.get();
        return -Math.log(value) - Math.log1p(-value);
    }

    @Override
    public RealScalarParam<? extends UnitInterval> getStateNode() {
        return this.parameter;
    }

}
