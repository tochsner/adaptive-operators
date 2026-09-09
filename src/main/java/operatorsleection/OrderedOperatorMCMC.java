package operatorsleection;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

/**
 * MCMC that applies the operators in a fixed cycle instead of drawing one at random each step.
 *
 * <p>The operators are taken in definition order with integer weights {@code w_k}, and the state
 * space is enlarged to {@code X x {1 ... C}} with {@code C = sum_k w_k}. The cycle is a fixed list
 * of segments, each naming an operator and a dwell length, and in extended state {@code (x, c)} only
 * the operator of the segment covering position {@code c} is used; the position then advances to
 * {@code (c mod C) + 1}. Dwelling on one operator for several steps samples its parameter block
 * conditioned on the rest, which is intended to help when the blocks moved by different operators
 * depend on each other non-trivially.
 *
 * <p>Without a cap there is one segment per operator, so operator {@code k} serves the positions
 * {@code c_{k-1} < c <= c_k} of the cumulative weights. With {@code maxDwell = D} the segments come
 * from a capped round robin: every operator starts with a budget of {@code w_k} steps, the operators
 * are swept in order and each one with budget left emits a segment of {@code min(D, budget)} steps,
 * and the sweep repeats until every budget is spent. Every operator still gets exactly {@code w_k}
 * steps per cycle, but no segment is longer than {@code D}, so a heavy operator is spread over
 * several visits rather than blocking the chain for its whole weight in one go.
 *
 * <p>This leaves the target unchanged. Every single Metropolis-Hastings step with operator
 * {@code k} is a kernel {@code P_k} that has the posterior {@code pi} as invariant distribution,
 * exactly as in the random-scan case, because the acceptance probability only depends on the
 * proposal of the operator that was chosen and not on how it was chosen. The extended chain moves
 * from {@code (x, c)} to {@code (x', c + 1)} with {@code x' ~ P_{k(c)}(x, .)}, so the product
 * {@code pi(x) * Uniform(c)} is invariant: a uniform position stays uniform since the position map
 * is a bijection, and given the position the {@code x} coordinate is pushed through a
 * {@code pi}-invariant kernel. The marginal of the extended stationary distribution on {@code X} is
 * {@code pi}, and the chain of {@code x} values visited by the loop is the marginal of the extended
 * chain. The extended chain is not reversible, but reversibility was never needed for correctness,
 * only invariance and irreducibility, and the latter holds as soon as one full cycle can reach any
 * state, which is the usual assumption made for the random scan as well. The argument only uses
 * that the operator at each position is fixed in advance, so it holds for any segment list.
 *
 * <p>Real-valued weights are rounded to integers after scaling by {@code 10^weightPrecision}. The
 * cycle position is not written to the state file, so a resumed run starts again at the first
 * segment. This does not affect the target, since the stationary distribution is invariant for any
 * starting position; it only restarts the deterministic scan at a fixed place.
 */
@Description("MCMC that cycles through its operators in definition order, dwelling on each " +
        "operator for as many steps as its integer weight prescribes, instead of picking a random " +
        "operator every step. A maximum dwell splits heavy operators into several shorter visits " +
        "per cycle.")
public class OrderedOperatorMCMC extends MCMC {

    public final Input<Integer> weightPrecisionInput = new Input<>("weightPrecision",
            "number of decimal places of the operator weights that are kept when rounding them " +
                    "to integer dwell times, so a weight of 0.25 needs a precision of at least 2",
            2);

    public final Input<Integer> maxDwellInput = new Input<>("maxDwell",
            "maximum number of consecutive steps spent on one operator; an operator whose integer " +
                    "weight exceeds this is visited several times per cycle, each visit at most this " +
                    "long, so that it still gets its full weight per cycle",
            20);

    /** One visit of the cycle to an operator: its index in definition order and the dwell length. */
    record Segment(int operatorIndex, int length) {
    }

    /** Operators in definition order. */
    private List<Operator> orderedOperators;

    /** The cycle as a list of segments. */
    private List<Segment> segments;

    /** Cumulative segment lengths; segment s serves the cycle positions ends[s-1] < c <= ends[s]. */
    private int[] segmentEnds;

    /** Cycle length C = sum_k w_k. */
    private int cycleLength;

    /** Current position c in the cycle, in 1..C. */
    private int position;

    /** Index of the segment that serves the current position. */
    private int segmentIndex;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        setUpCycle(operatorsInput.get(), weightPrecisionInput.get(), maxDwellInput.get());
    }

    /**
     * Fixes the operator order and the segment list, and rewinds the cycle to its start.
     */
    void setUpCycle(List<Operator> operators, int weightPrecision, int maxDwell) {
        orderedOperators = new ArrayList<>(operators);
        segments = cappedRoundRobin(integerWeights(operators, weightPrecision), maxDwell);

        segmentEnds = new int[segments.size()];
        int total = 0;
        for (int s = 0; s < segments.size(); s++) {
            total += segments.get(s).length();
            segmentEnds[s] = total;
        }

        cycleLength = total;
        position = 1;
        segmentIndex = 0;
    }

    /**
     * Turns the real-valued operator weights into integer weights.
     *
     * <p>Every weight is scaled by {@code 10^weightPrecision} and rounded to the nearest integer.
     * A weight that rounds to zero would make its operator unreachable, which is an error rather
     * than a silent drop, since the deterministic scan may then fail to be irreducible.
     */
    static int[] integerWeights(List<Operator> operators, int weightPrecision) {
        if (operators.isEmpty()) {
            throw new IllegalArgumentException("At least one operator is required to build the operator cycle.");
        }
        if (weightPrecision < 0) {
            throw new IllegalArgumentException("weightPrecision must not be negative.");
        }

        double scale = Math.pow(10, weightPrecision);
        int[] weights = new int[operators.size()];
        long total = 0;

        for (int k = 0; k < operators.size(); k++) {
            Operator operator = operators.get(k);
            long weight = Math.round(operator.getWeight() * scale);
            if (weight <= 0) {
                throw new IllegalArgumentException("Operator " + operator.getID() + " has weight " + operator.getWeight()
                        + ", which rounds to zero at weightPrecision " + weightPrecision
                        + ". Increase weightPrecision or the weight.");
            }
            total += weight;
            if (total > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("The summed integer operator weights exceed " + Integer.MAX_VALUE
                        + ". Reduce weightPrecision or the weights.");
            }
            weights[k] = (int) weight;
        }

        return weights;
    }

    /**
     * Builds the segment list by sweeping the operators in order and letting each one with budget
     * left dwell for at most {@code maxDwell} steps, until every operator has spent its weight.
     *
     * <p>Sweeping rather than emitting all of an operator's segments at once is what spreads a
     * heavy operator across the cycle. The segments of one sweep appear in definition order, so
     * with a cap of at least the largest weight there is a single sweep and exactly one segment
     * per operator.
     */
    static List<Segment> cappedRoundRobin(int[] weights, int maxDwell) {
        if (maxDwell < 1) {
            throw new IllegalArgumentException("maxDwell must be at least 1.");
        }

        int[] remaining = weights.clone();
        List<Segment> segments = new ArrayList<>();

        boolean anyLeft = true;
        while (anyLeft) {
            anyLeft = false;
            for (int k = 0; k < remaining.length; k++) {
                if (remaining[k] == 0) {
                    continue;
                }
                int length = Math.min(maxDwell, remaining[k]);
                segments.add(new Segment(k, length));
                remaining[k] -= length;
                anyLeft |= remaining[k] > 0;
            }
        }

        return segments;
    }

    /**
     * Returns the operator serving the current cycle position and advances the position by one.
     */
    Operator nextOperator() {
        Operator operator = orderedOperators.get(segments.get(segmentIndex).operatorIndex());

        // advance c' = (c mod C) + 1 and move to the next segment once this one is used up
        position = (position % cycleLength) + 1;
        if (position == 1) {
            segmentIndex = 0;
        } else if (position > segmentEnds[segmentIndex]) {
            segmentIndex++;
        }

        return operator;
    }

    /** Current position c in the cycle, in 1..C. */
    int getPosition() {
        return position;
    }

    /** The cycle as a list of segments. */
    List<Segment> getSegments() {
        return segments;
    }

    /**
     * One Metropolis-Hastings step, identical to the one of {@link MCMC} except that the operator
     * comes from the cycle instead of from the operator schedule.
     */
    @Override
    protected Operator propagateState(final long sampleNr) {
        state.store(sampleNr);

        final Operator operator = nextOperator();

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
