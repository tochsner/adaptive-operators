package mala;

/**
 * Shared mechanics of a single MALA-style proposal: take the deterministic forward step,
 * add noise, apply the proposal to the state, and return the resulting log Hastings ratio.
 * The per-operator variation lives entirely in the {@link GaussianProposalKernel} and the
 * {@link StateUpdate}.
 */
public final class GaussianProposal {

    private GaussianProposal() {
    }

    /**
     * Applies the proposed mutable values to the state (adapters, cube, ...).
     */
    @FunctionalInterface
    public interface StateUpdate {
        /**
         * @return the combined Jacobian and transition contribution
         *         (oldLogJacobian - newLogJacobian + transitionCorrection), or
         *         {@link Double#NEGATIVE_INFINITY} to reject the proposal.
         */
        double apply(double[] proposedMutable);
    }

    /**
     * The outcome of a proposal: the log Hastings ratio plus the two forward steps, so that
     * callers such as Fisher adaptation can read the gradients carried as auxiliary data.
     */
    public record Result(double logHastingsRatio, GaussianProposalKernel.Step forward,
                         GaussianProposalKernel.Step reverse) {
    }

    public static Result propose(double[] currentMutable, GaussianProposalKernel kernel, StateUpdate update) {
        GaussianProposalKernel.Step forward = kernel.forwardStep(currentMutable);
        if (forward == null) return reject();

        double[] noise = kernel.sampleNoise();
        double[] proposed = new double[currentMutable.length];
        for (int i = 0; i < proposed.length; i++) {
            proposed[i] = forward.mean()[i] + noise[i];
            if (!Double.isFinite(proposed[i])) return reject();
        }

        double jacobianAndTransition = update.apply(proposed);
        if (!Double.isFinite(jacobianAndTransition)) return reject();

        GaussianProposalKernel.Step reverse = kernel.forwardStep(proposed);
        if (reverse == null) return reject();

        double ratio = kernel.logProposalRatio(currentMutable, forward, proposed, reverse);
        if (!Double.isFinite(ratio)) return reject();

        return new Result(ratio + jacobianAndTransition, forward, reverse);
    }

    private static Result reject() {
        return new Result(Double.NEGATIVE_INFINITY, null, null);
    }
}
