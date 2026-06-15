package transforms;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.spec.domain.NonNegativeInt;
import beast.base.spec.inference.parameter.IntSimplexParam;

public class SimplexRoundTransform extends BEASTObject implements RealVectorTransform<IntSimplexParam<? extends NonNegativeInt>> {

    public final Input<IntSimplexParam<? extends NonNegativeInt>> parameterInput = new Input<>("parameter", "");
    public final Input<Integer> denominatorIndexInput = new Input<>("denominatorIndex",
            "component to use as the ALR denominator; negative values use the last component", -1);

    private IntSimplexParam<? extends NonNegativeInt> parameter;
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
        int denominator = this.parameter.get(this.denominatorIndex);

        if (denominator <= 0) {
            throw new IllegalStateException("Cannot apply ALR transform when the denominator component is non-positive");
        }

        int transformedIndex = 0;
        for (int i = 0; i < this.parameter.size(); i++) {
            if (i == this.denominatorIndex) {
                continue;
            }

            int numerator = this.parameter.get(i);

            if (numerator <= 0) {
                throw new IllegalStateException("Cannot apply ALR transform when a simplex component is non-positive");
            }

            transformed[transformedIndex++] = Math.log((double) numerator / denominator);
        }

        return transformed;
    }

    @Override
    public void set(Double[] value) {
        if (value.length != getDimension()) {
            throw new IllegalArgumentException("Expected " + getDimension()
                    + " ALR coordinates, but got " + value.length);
        }

        int expectedSum = this.parameter.expectedSum();

        if (expectedSum < 0) {
            throw new IllegalStateException("Cannot round simplex coordinates to a negative expected sum");
        }

        double[] probabilities = getInverseAlrProbabilities(value);
        int[] rounded = roundToExpectedSum(probabilities, expectedSum);

        for (int i = 0; i < rounded.length; i++) {
            this.parameter.set(i, rounded[i]);
        }
    }

    private double[] getInverseAlrProbabilities(Double[] value) {
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

        double[] probabilities = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp(logits[i] - max) / sum;
        }

        return probabilities;
    }

    private static int[] roundToExpectedSum(double[] probabilities, int expectedSum) {
        int[] rounded = new int[probabilities.length];
        double[] remainders = new double[probabilities.length];
        int roundedSum = 0;

        for (int i = 0; i < probabilities.length; i++) {
            double scaled = probabilities[i] * expectedSum;
            rounded[i] = (int) Math.floor(scaled);
            remainders[i] = scaled - rounded[i];
            roundedSum += rounded[i];
        }

        int remaining = expectedSum - roundedSum;
        while (remaining > 0) {
            int bestIndex = 0;

            for (int i = 1; i < remainders.length; i++) {
                if (remainders[i] > remainders[bestIndex]) {
                    bestIndex = i;
                }
            }

            rounded[bestIndex]++;
            remainders[bestIndex] = -1.0;
            remaining--;
        }

        return rounded;
    }

    @Override
    public double getLogJacobianCorrection() {
        double expectedSum = this.parameter.expectedSum();
        double logCorrection = 0.0;

        if (expectedSum <= 0.0) {
            throw new IllegalStateException("Cannot compute ALR Jacobian when expectedSum is non-positive");
        }

        for (int i = 0; i < this.parameter.size(); i++) {
            int component = this.parameter.get(i);

            if (component <= 0) {
                throw new IllegalStateException("Cannot compute ALR Jacobian when a simplex component is non-positive");
            }

            logCorrection -= Math.log(component / expectedSum);
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
    public IntSimplexParam<? extends NonNegativeInt> getStateNode() {
        return this.parameter;
    }

}
