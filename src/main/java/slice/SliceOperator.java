package slice;

import beast.base.inference.Operator;
import beast.base.inference.State;

import java.util.function.Supplier;

public abstract class SliceOperator extends Operator {

    public abstract double proposal(Supplier<Double> computeCurrentLogLikelihood, State state);

    /**
     * Marks everything a rejected proposal may have left stale in the calculation cache. Called by
     * JumpMCMC after the state nodes have been restored and before the posterior is recalculated.
     *
     * <p>A slice proposal evaluates the posterior many times, so the cache cannot be restored and has
     * to be recalculated. The default marks everything dirty, which is always correct. Operators that
     * know which part of the state their proposal touched can mark only that and make rejections cheaper.
     */
    public void markDirtyAfterRestore(State state) {
        state.setEverythingDirty(true);
    }

    @Override
    public double proposal() {
        throw new UnsupportedOperationException();
    }

}
