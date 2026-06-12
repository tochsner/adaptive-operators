package adaptiveoperators;

/**
 * Adapts a {@link ConditionalSampler} to the {@link GradientModel} interface so a fitted
 * distribution (e.g. {@link MultivariateNormalSampler}) can stand in for a neural network as
 * the gradient source of a MALA-style operator. Each recorded input vector is modeled as the
 * sampler's values with no conditions; the scalar target is ignored, because the gradient is
 * taken from the fitted density of the inputs themselves rather than from a learned input ->
 * target mapping.
 */
public class ConditionalSamplerGradientModel implements GradientModel {

    private static final double[] NO_CONDITIONS = new double[]{};

    private final ConditionalSampler sampler;

    public ConditionalSamplerGradientModel(ConditionalSampler sampler) {
        this.sampler = sampler;
    }

    @Override
    public void record(double[] inputs, double[] output) {
        this.sampler.record(NO_CONDITIONS, inputs);
    }

    @Override
    public double[] getGradient(double[] inputs) {
        return this.sampler.getGradient(inputs);
    }
}
