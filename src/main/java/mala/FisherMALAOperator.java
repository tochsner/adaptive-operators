package mala;

import beast.base.core.Input;
import beast.base.util.Randomizer;

import java.util.Arrays;

/**
 * MALA operator that learns the preconditioner A = R R^T online via Fisher adaptation
 * (Titsias 2023, arXiv 2305.14442, Algorithm 1). It plugs a Fisher kernel into the shared
 * {@link GaussianProposal} mechanics: the forward step is the preconditioned gradient drift,
 * the noise is R-correlated, and the proposal ratio uses the closed form of Proposition 1
 * (no matrix inverse). The square root R is adapted with rank-one updates driven by
 * s = sqrt(alpha) (grad log pi(y) - grad log pi(x)), and the step size follows
 * sigma^2 *= 1 + rho (alpha - targetAcceptance). The true acceptance probability alpha is
 * obtained from BEAST's optimize(logAlpha) callback, which receives the full log MH ratio
 * after every proposal.
 */
public class FisherMALAOperator extends MALAOperator {

    public final Input<Double> lambdaInput = new Input<>(
            "lambda",
            "regularization strength of the initial preconditioner (R is initialized to (1/sqrt(lambda)) I)",
            10.0);
    public final Input<Double> rhoInput = new Input<>(
            "rho",
            "learning rate of the step size adaptation",
            0.015);
    public final Input<Double> targetAcceptanceInput = new Input<>(
            "targetAcceptance",
            "target acceptance probability of the step size adaptation",
            0.574);
    public final Input<Double> initialStepSizeInput = new Input<>(
            "initialStepSize",
            "initial (unnormalized) step size sigma^2",
            1.0);

    private FisherPreconditioner preconditioner;

    // proposal gradients consumed by optimize() to form the adaptation signal
    private double[] stashedOldGradient;
    private double[] stashedProposedGradient;
    private boolean stashValid = false;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.preconditioner = new FisherPreconditioner(
                this.totalNumMutable,
                this.lambdaInput.get(),
                this.rhoInput.get(),
                this.targetAcceptanceInput.get(),
                this.initialStepSizeInput.get());
    }

    @Override
    public double proposal() {
        this.stashValid = false;

        if (this.count < this.JUST_TRAINING) {
            return this.proposeAlternativeOperator();
        }

        this.refreshAdapters();

        int nodeId = this.getNodeId();

        double[] oldState = this.getState(nodeId);
        double[] oldMutable = Arrays.copyOf(oldState, this.totalNumMutable);
        double[] immutable = Arrays.copyOfRange(oldState, this.totalNumMutable, oldState.length);

        GaussianProposalKernel kernel = new Kernel(immutable);

        GaussianProposal.Result result = GaussianProposal.propose(
                oldMutable,
                kernel,
                proposed -> this.applyAdapters(proposed, nodeId));

        // stash the gradients (carried as auxiliary data) so optimize() can form the signal
        if (result.forward() != null && result.reverse() != null) {
            this.stashedOldGradient = result.forward().auxiliary();
            this.stashedProposedGradient = result.reverse().auxiliary();
            this.stashValid = true;
        }

        return result.logHastingsRatio();
    }

    @Override
    public void optimize(double logAlpha) {
        if (this.count < this.JUST_TRAINING) return;
        if (!this.stashValid) return;
        this.stashValid = false;

        double alpha = Math.min(1.0, Math.exp(logAlpha));
        if (!Double.isFinite(alpha)) return;

        double[] signal = new double[this.totalNumMutable];
        for (int i = 0; i < signal.length; i++) {
            signal[i] = Math.sqrt(alpha) * (this.stashedProposedGradient[i] - this.stashedOldGradient[i]);
        }

        this.preconditioner.updateSquareRoot(signal);
        this.preconditioner.updateStepSize(alpha);
    }

    @Override
    public double getCoercableParameterValue() {
        return this.preconditioner.getSigma2();
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.preconditioner.setSigma2(value);
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return this.targetAcceptanceInput.get();
    }

    /**
     * Fisher adaptive MALA kernel: drift (sigma2R / 2) R (R^T grad), R-correlated noise, and
     * the Proposition 1 closed-form proposal ratio. The gradient at each point is carried as
     * the step's auxiliary data so the operator can build the adaptation signal.
     */
    private final class Kernel implements GaussianProposalKernel {

        private final double[] immutable;
        private final double sigma2R;

        private Kernel(double[] immutable) {
            this.immutable = immutable;
            this.sigma2R = preconditioner.sigma2R();
        }

        @Override
        public Step forwardStep(double[] point) {
            double[] gradient = Arrays.copyOf(gradientModel.getGradient(assemble(point, this.immutable)), totalNumMutable);
            if (!allFinite(gradient)) return null;

            double[] drift = preconditioner.driftIncrement(gradient, this.sigma2R);
            double[] mean = new double[point.length];
            for (int i = 0; i < point.length; i++) {
                mean[i] = point[i] + drift[i];
            }

            return new Step(mean, gradient);
        }

        @Override
        public double[] sampleNoise() {
            double[] eta = new double[totalNumMutable];
            for (int i = 0; i < eta.length; i++) {
                eta[i] = Randomizer.nextGaussian();
            }

            return preconditioner.sampleNoise(eta, this.sigma2R);
        }

        @Override
        public double logProposalRatio(double[] oldPoint, Step forward, double[] newPoint, Step reverse) {
            return preconditioner.hastings(
                    oldPoint, newPoint, forward.auxiliary(), reverse.auxiliary(), this.sigma2R);
        }
    }

}
