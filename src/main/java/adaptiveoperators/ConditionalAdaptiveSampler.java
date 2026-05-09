package adaptiveoperators;

public abstract class ConditionalAdaptiveSampler {

    protected final int numConditions;
    protected final int numValues;

    public ConditionalAdaptiveSampler(int numConditions, int numValues) {
        this.numConditions = numConditions;
        this.numValues = numValues;
    }

    public abstract void record(double[] conditions, double[] values);
    public abstract double[] sampleConditionally(double[] conditions);

}
