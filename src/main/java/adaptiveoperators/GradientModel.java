package adaptiveoperators;

/**
 * A trainable approximation of a scalar log-target that can report the gradient of that
 * approximation with respect to its inputs. MALA-style operators use it as an interchangeable
 * source of (approximate) gradients: train it online with {@link #record} and query the
 * gradient at a point with {@link #getGradient}. Implementations include a neural network
 * ({@code ml.MLP}) and a fitted multivariate normal wrapped by
 * {@link ConditionalSamplerGradientModel}.
 */
public interface GradientModel {

    /**
     * Records a training example mapping {@code inputs} to the scalar target {@code output}.
     */
    void record(double[] inputs, double[] output);

    /**
     * Returns the gradient of the approximated log-target at {@code inputs}.
     */
    double[] getGradient(double[] inputs);
}
