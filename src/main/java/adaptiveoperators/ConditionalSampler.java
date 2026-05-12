package adaptiveoperators;

public abstract class ConditionalSampler {

    protected final int numConditions;
    protected final int numValues;

    public ConditionalSampler(int numConditions, int numValues) {
        this.numConditions = numConditions;
        this.numValues = numValues;
    }

    public abstract void record(double[] conditions, double[] values);
    public abstract double[] sampleConditionally(double[] conditions);
    public abstract double logDensity(double[] conditions, double[] values);

}
