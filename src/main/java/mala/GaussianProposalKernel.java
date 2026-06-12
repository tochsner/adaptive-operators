package mala;

/**
 * A Gaussian proposal of the form N(point + drift(point), Sigma) used by the MALA-style
 * operators. Every operator follows the same skeleton — current state, deterministic
 * forward step, additive noise, Hastings ratio — and differs only in how the forward step
 * (the drift) and the noise are generated. Those differences are captured here.
 */
public interface GaussianProposalKernel {

    /**
     * A forward step: the proposal mean for a point together with any kernel-specific
     * auxiliary data (e.g. the gradient) needed later to evaluate the proposal density.
     */
    record Step(double[] mean, double[] auxiliary) {
        public Step(double[] mean) {
            this(mean, null);
        }
    }

    /**
     * Computes the deterministic forward step for {@code point}. Returns {@code null} to
     * signal an invalid proposal (e.g. a non-finite gradient), which is rejected.
     */
    Step forwardStep(double[] point);

    /**
     * Draws a zero-mean noise vector with this kernel's covariance, to be added to a mean.
     */
    double[] sampleNoise();

    /**
     * Returns the log proposal ratio log q(oldPoint | newPoint) - log q(newPoint | oldPoint),
     * where {@code forward} is the step taken at {@code oldPoint} and {@code reverse} the step
     * taken at {@code newPoint}.
     */
    double logProposalRatio(double[] oldPoint, Step forward, double[] newPoint, Step reverse);
}
