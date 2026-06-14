package mcmc;

import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import slice.SliceOperator;

import java.util.function.Supplier;

/**
 * MCMC driver for operators that run an inner optimization, evaluating the posterior many
 * times within a single proposal (see {@link mala.LargeJumpMALAOperator}).
 *
 * <p>BEAST's store/restore cache is single-buffered: it can represent the current state plus
 * exactly one stored state, and assumes a single store -> mutate -> calculate -> accept/restore
 * transaction per step. An inner optimization visits many intermediate states, so the pre-proposal
 * cache cannot survive the walk and {@code restoreCalculationNodes()} becomes unrecoverable. This
 * driver therefore bypasses the incremental cache for {@link SliceOperator}s: the likelihood
 * supplier recomputes the current posterior without snapshotting, and a rejected proposal is
 * recovered by a full recalculation rather than by restoring the corrupted cache. Non-slice
 * operators keep the standard, efficient incremental path.
 */
public class JumpMCMC extends MCMC {

    protected Operator propagateState(final long sampleNr) {
        state.store(sampleNr);

        final Operator operator = operatorSchedule.selectOperator();

        if (operator instanceof SliceOperator sliceOperator) {
            return this.propagateJumpOperator(sampleNr, operator, sliceOperator);
        }

        return this.propagateStandardOperator(sampleNr, operator);
    }

    /**
     * Robust path for operators that evaluate the posterior repeatedly inside a single proposal.
     * The cache is never trusted for restoration; rejected proposals are recovered by a full
     * recomputation against the restored state nodes.
     */
    private Operator propagateJumpOperator(final long sampleNr, final Operator operator, final SliceOperator sliceOperator) {
        // cache-bypassing supplier: mark dirty and recompute so the returned value reflects the
        // current state, but never call storeCalculationNodes() (which would overwrite the single
        // stored buffer with an intermediate walk state) and never touch the state-node backups
        // (which restore() still needs to be correct)
        Supplier<Double> computeCurrentLogLikelihood = () -> {
            state.checkCalculationNodesDirtiness();
            return posterior.calculateLogP();
        };

        final double logHastingsRatio = sliceOperator.proposal(computeCurrentLogLikelihood, this.state);

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
            // recompute at the proposed state without storeCalculationNodes() or setEverythingDirty(),
            // so the per-node single backups stay intact for a possible restore
            state.checkCalculationNodesDirtiness();
            newLogLikelihood = posterior.calculateLogP();

            logAlpha = newLogLikelihood - oldLogLikelihood + logHastingsRatio;

            if (logAlpha >= 0 || (logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(logAlpha))) {
                // accept: the current buffers already reflect the proposed state, so just commit them
                oldLogLikelihood = newLogLikelihood;
                state.acceptCalculationNodes();

                if (sampleNr >= 0) {
                    operator.accept();
                }
            } else {
                // reject: restore the state nodes from their pre-proposal backups, then resync the
                // calculation cache from scratch since the inner walk left it unrecoverable
                if (sampleNr >= 0) {
                    operator.reject(newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                }
                state.restore();
                this.robustlyRecalculate();
            }
            state.setEverythingDirty(false);
        } else {
            logAlpha = Double.NEGATIVE_INFINITY;
            // operation failed
            if (sampleNr >= 0) {
                operator.reject(-2);
            }
            state.restore();
            this.robustlyRecalculate();
            state.setEverythingDirty(false);
        }
        log(sampleNr);
        return operator;
    }

    /**
     * Recompute the whole posterior from scratch and leave the cache clean and consistent with the
     * current state. Used to recover after a rejected jump proposal, whose inner walk corrupts the
     * incremental cache beyond what {@code restoreCalculationNodes()} can undo. Must only be called
     * after {@code state.restore()}, never before, because {@code setEverythingDirty(true)} marks
     * every node dirty without backing it up.
     */
    private void robustlyRecalculate() {
        state.setEverythingDirty(true);
        state.checkCalculationNodesDirtiness();
        posterior.calculateLogP();
        state.setEverythingDirty(false);
        state.acceptCalculationNodes();
    }

    /** Standard incremental Metropolis-Hastings path for ordinary operators. */
    private Operator propagateStandardOperator(final long sampleNr, final Operator operator) {
        final double logHastingsRatio = operator.proposal();

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {

            if (operator.requiresStateInitialisation()) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
            }

            newLogLikelihood = posterior.calculateLogP();

            logAlpha = newLogLikelihood - oldLogLikelihood + logHastingsRatio;

            if (logAlpha >= 0 || (logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(logAlpha))) {
                // accept
                oldLogLikelihood = newLogLikelihood;
                state.acceptCalculationNodes();

                if (sampleNr >= 0) {
                    operator.accept();
                }
            } else {
                // reject
                if (sampleNr >= 0) {
                    operator.reject(newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                }
                state.restore();
                state.restoreCalculationNodes();
            }
            state.setEverythingDirty(false);
        } else {
            logAlpha = Double.NEGATIVE_INFINITY;
            // operation failed
            if (sampleNr >= 0) {
                operator.reject(-2);
            }
            state.restore();
            if (!operator.requiresStateInitialisation()) {
                state.setEverythingDirty(false);
                state.restoreCalculationNodes();
            }
        }
        log(sampleNr);
        return operator;
    }

}
