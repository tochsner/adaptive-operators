package mala;

/**
 * Convenience base for kernels whose proposal ratio is the difference of two explicit
 * Gaussian log densities (the common case). Subclasses only supply the deterministic
 * forward step, the noise, and a log density. Kernels with a closed-form ratio that avoids
 * forming the density (e.g. Fisher adaptive MALA) implement {@link GaussianProposalKernel}
 * directly instead.
 */
public abstract class AbstractDensityKernel implements GaussianProposalKernel {

    /** Deterministic forward step: the proposal mean for {@code point}. */
    public abstract double[] mean(double[] point);

    @Override
    public abstract double[] sampleNoise();

    /** Log density of {@code point} under the Gaussian centered at {@code mean}. */
    public abstract double logDensity(double[] point, double[] mean);

    @Override
    public Step forwardStep(double[] point) {
        return new Step(this.mean(point));
    }

    @Override
    public double logProposalRatio(double[] oldPoint, Step forward, double[] newPoint, Step reverse) {
        double reverseLogDensity = this.logDensity(oldPoint, reverse.mean());
        double forwardLogDensity = this.logDensity(newPoint, forward.mean());
        return reverseLogDensity - forwardLogDensity;
    }
}
