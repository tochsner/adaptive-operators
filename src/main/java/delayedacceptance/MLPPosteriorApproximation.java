package delayedacceptance;

import ml.MLP;

import java.util.Arrays;

public class MLPPosteriorApproximation extends PosteriorApproximation {

    int BURN_IN = 100_000;

    private static final int OUTPUT_DIM = 1;

    private MLP mlp;
    private int registeredFiniteLogPosteriorDifferences;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.mlp = new MLP(this.numValues * 3, OUTPUT_DIM);
        this.registeredFiniteLogPosteriorDifferences = 0;
    }

    @Override
    public double approximateLogPosteriorDifference(double[] previousValues) {
        double[] currentValues = this.getCurrentValues();

        if (!isValidTransition(previousValues, currentValues)) {
            return Double.NEGATIVE_INFINITY;
        }

        return this.mlp.runInference(transitionValues(previousValues, currentValues))[0];
    }

    @Override
    public void registerLogPosteriorDifference(double logPosteriorDifference, double[] previousValues) {
        double[] currentValues = this.getCurrentValues();

        if (!isValidTransition(previousValues, currentValues)) return;
        if (!Double.isFinite(logPosteriorDifference)) return;

        this.registeredFiniteLogPosteriorDifferences++;
        if (this.registeredFiniteLogPosteriorDifferences > this.BURN_IN) return;

        this.mlp.record(
                transitionValues(previousValues, currentValues),
                new double[] {logPosteriorDifference}
        );
    }

    @Override
    public boolean isReady() {
        return this.registeredFiniteLogPosteriorDifferences >= this.BURN_IN;
    }

    private boolean isValidTransition(double[] previousValues, double[] currentValues) {
        return currentValues.length == previousValues.length
                && Arrays.stream(currentValues).allMatch(Double::isFinite)
                && Arrays.stream(previousValues).allMatch(Double::isFinite);
    }

    private double[] transitionValues(double[] previousValues, double[] currentValues) {
        double[] values = new double[previousValues.length + currentValues.length + currentValues.length];

        for (int i = 0; i < previousValues.length; i++) {
            values[i] = previousValues[i];
        }
        for (int i = 0; i < currentValues.length; i++) {
            values[previousValues.length + i] = currentValues[i];
            values[previousValues.length + currentValues.length + i] = currentValues[i] - previousValues[i];
        }

        return values;
    }
}
