package adaptiveoperators;

public class GaussianMixtureSampler extends ConditionalSampler {

    public GaussianMixtureSampler(int numConditions, int numValues) {
        super(numConditions, numValues);
    }

    @Override
    public void record(double[] conditions, double[] values) {
        // TODO
    }

    @Override
    public double[] sampleConditionally(double[] conditions) {
        // TODO
    }

    @Override
    public double logDensity(double[] conditions, double[] values) {
        // TODO
    }
}
