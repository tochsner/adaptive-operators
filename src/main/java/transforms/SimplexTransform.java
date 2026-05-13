package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.inference.parameter.SimplexParam;

public class SimplexTransform extends BEASTObject implements RealVectorTransform<SimplexParam> {

    public final Input<SimplexParam> parameterInput = new Input<>("parameter", "");
    public final Input<Integer> denominatorIndexInput = new Input<>("denominatorIndex",
            "component to use as the ALR denominator; negative values use the last component", -1);

    private SimplexParam parameter;
    private int denominatorIndex;

    @Override
    public void initAndValidate() {
        this.parameter = this.parameterInput.get();
        this.denominatorIndex = this.denominatorIndexInput.get();

        if (this.parameter.size() < 2) {
            throw new IllegalArgumentException("ALR transform requires a simplex with at least two components");
        }

        if (this.denominatorIndex < 0) {
            this.denominatorIndex = this.parameter.size() - 1;
        }

        if (this.denominatorIndex >= this.parameter.size()) {
            throw new IllegalArgumentException("denominatorIndex must identify a simplex component");
        }
    }

    @Override
    public Double[] get() {
        Double[] transformed = new Double[getDimension()];
        double denominator = this.parameter.get(this.denominatorIndex);

        if (denominator <= 0.0) {
            throw new IllegalStateException("Cannot apply ALR transform when the denominator component is non-positive");
        }

        int transformedIndex = 0;
        for (int i = 0; i < this.parameter.size(); i++) {
            if (i == this.denominatorIndex) {
                continue;
            }

            double numerator = this.parameter.get(i);

            if (numerator <= 0.0) {
                throw new IllegalStateException("Cannot apply ALR transform when a simplex component is non-positive");
            }

            transformed[transformedIndex++] = Math.log(numerator / denominator);
        }

        return transformed;
    }

    @Override
    public void set(Double[] value) {
        if (value.length != getDimension()) {
            throw new IllegalArgumentException("Expected " + getDimension()
                    + " ALR coordinates, but got " + value.length);
        }

        double[] logits = new double[this.parameter.size()];
        int transformedIndex = 0;

        for (int i = 0; i < logits.length; i++) {
            if (i == this.denominatorIndex) {
                continue;
            }

            logits[i] = value[transformedIndex++];
        }

        double max = 0.0;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }

        double sum = 0.0;
        for (double logit : logits) {
            sum += Math.exp(logit - max);
        }

        for (int i = 0; i < logits.length; i++) {
            this.parameter.set(i, Math.exp(logits[i] - max) / sum);
        }
    }

    @Override
    public double getLogJacobianCorrection() {
        double logCorrection = 0.0;

        for (int i = 0; i < this.parameter.size(); i++) {
            double component = this.parameter.get(i);

            if (component <= 0.0) {
                throw new IllegalStateException("Cannot compute ALR Jacobian when a simplex component is non-positive");
            }

            logCorrection -= Math.log(component);
        }

        return logCorrection;
    }

    @Override
    public int getDimension() {
        return this.parameter.size() - 1;
    }

    @Override
    public double getLogJacobianCorrection(int index) {
        throw new RuntimeException("Simplex cannot be used element-wise");
    }

    @Override
    public SimplexParam getStateNode() {
        return this.parameter;
    }

}
