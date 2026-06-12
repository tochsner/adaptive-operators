package adaptiveoperators;

public abstract class ConditionalSampler {

    protected final int numConditions;
    protected final int numValues;

    public ConditionalSampler(int numConditions, int numValues) {
        this.numConditions = numConditions;
        this.numValues = numValues;
    }

    public abstract void record(double[] conditions, double[] values);
    public abstract double[] sampleConditionally(double[] conditions, double scaleFactor);
    public abstract double logDensity(double[] conditions, double[] values, double scaleFactor);

    /**
     * Returns the gradient of the fitted log-density with respect to {@code inputs}. This is
     * optional: the default implementation is unsupported, and samplers that can provide a
     * gradient override it.
     */
    public double[] getGradient(double[] inputs) {
        throw new UnsupportedOperationException(
                "getGradient is not implemented for " + this.getClass().getSimpleName());
    }

}
