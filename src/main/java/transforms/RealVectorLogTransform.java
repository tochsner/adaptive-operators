package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.NonNegativeReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealVectorParam;

public class RealVectorLogTransform extends BEASTObject implements RealVectorTransform<RealVectorParam<? extends Real>> {

    public final Input<? extends RealVectorParam<? extends NonNegativeReal>> parameterInput = new Input<>("parameter", "");

    private RealVectorParam<? extends NonNegativeReal> parameter;

    @Override
    public void initAndValidate() {
        this.parameter = this.parameterInput.get();
    }


    @Override
    public Double[] get() {
        Double[] transformed = new Double[this.parameter.size()];
        for (int i = 0; i < this.parameter.size(); i++) {
            transformed[i] = Math.log(this.parameter.get(i));
        }
        return transformed;
    }

    @Override
    public void set(Double[] value) {
        for (int i = 0; i < this.parameter.size(); i++) {
            this.parameter.set(i, Math.exp(value[i]));
        }
    }

    @Override
    public double getLogJacobianCorrection() {
        double logCorrection = 0.0;

        for (int i = 0; i < this.parameter.size(); i++) {
            logCorrection -= Math.log(this.parameter.get(i));
        }

        return logCorrection;
    }

    @Override
    public double getLogJacobianCorrection(int index) {
        return -Math.log(this.parameter.get(index));
    }

    @Override
    public RealVectorParam<? extends Real> getStateNode() {
        return this.parameter;
    }

}
