package multipletry;

import adapters.Adapter;
import adapters.AdapterGenerator;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.State;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import slice.SliceOperator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * A multiple-try Metropolis kernel over the concatenated mutable vector of a set of adapters.
 *
 * <p>Every iteration fires {@code numTries} candidates at once, each perturbing all coordinates
 * of the mutable vector, and selects among them with the Barker weight
 * {@code w(y, x) = pi(y) / (pi(x) + pi(y))}. This is the Liu-Liang-Wong multiple-try weight for
 * the symmetric choice {@code lambda(x, y) = 2 / (pi(x) T(x, y) + pi(y) T(y, x))}. Because the
 * weight saturates at one, a single candidate landing on a narrow spike cannot monopolise the
 * selection the way it does under the plain weight {@code w = pi(y)}.
 *
 * <p>The proposal is a Gaussian centred on the current point with covariance
 * {@code s^2 diag(sigma^2)}, where {@code sigma} holds the learned per-coordinate standard
 * deviations and {@code s} is a single globally adapted scale. Both are frozen for the duration
 * of an iteration, which makes the proposal symmetric and is what lets every weight below be
 * computed without ever evaluating the proposal density.
 *
 * <p>The correction is completed by sampling a reverse neighbourhood around the selected
 * candidate: {@code numTries - 1} points drawn from the same kernel centred on the selection,
 * plus the current point itself. The acceptance probability is then the ratio of the two weight
 * sums. All of this costs {@code 2 * numTries + 1} posterior evaluations per iteration, so the
 * operator wants a low weight in the schedule relative to cheap moves.
 *
 * <p>Densities are taken in the mutable coordinates, which is the space the Gaussian lives in,
 * so each one is the posterior less the adapters' log Jacobian correction. The three operators
 * in the {@code slice} package drop that term; this one keeps it.
 *
 * <p>The standard deviations are keyed on {@link Adapter#getMutableKeys(int)} rather than on the
 * coordinate's position. This matters for adapters that reshuffle their coordinates:
 * {@code CubeAdapter} redraws its cube on every {@code refresh()}, so position {@code i} spans a
 * different pair of taxa each iteration. Keying on position blends every pair's spread into one
 * number, which leaves the global scale to be tuned down to whatever the tightest pair tolerates
 * and makes every move far too local. Keying on the pair gives each its own scale.
 *
 * <p>Candidates are not all drawn at the same scale. Each one picks a rung uniformly from a
 * geometric ladder centred on the adapted scale, so the fine rungs refine within the current mode
 * while the coarse ones attempt to cross into another; the weighting discards whichever fail at no
 * extra cost. A uniform mixture of symmetric kernels is itself symmetric, so this leaves the
 * weights in the form above. The rung is drawn per candidate rather than assigned by index, which
 * is what keeps the candidates independent draws from one common kernel.
 *
 * <p>The ladder is not free, and whether it pays depends on how far apart the modes are. On a
 * two-component target whose modes the adapted scale can already span, the coarse rungs overshoot
 * and waste tries: measured over fifty thousand iterations on modes ten apart, three rungs crossed
 * 2,647 times against 9,724 for a single scale. On modes thirty apart, where a single scale is
 * genuinely trapped, the same ladder first crossed at iteration 34 and crossed 1,462 times against
 * iteration 2,073 and 152 crossings. Widening it further does not keep helping; five rungs at
 * spacing eight managed only 603. Watch the per-rung breakdown in the performance report and widen
 * the ladder only while the coarsest rung is still winning moves.
 *
 * <p>The per-coordinate standard deviations remain poorly suited to a multimodal target: once the
 * chain has seen two modes they converge towards the spread of the mixture rather than the spread
 * within a mode, so the base scale inflates and the fine rungs stop being fine. The ladder softens
 * this, since a small rung recovers a local move, but the estimator underneath is still mismatched.
 * Adapting per region of the state space is the real fix and would break the symmetry above.
 *
 * <p>One limitation is worth stating: the value returned by
 * {@link Adapter#update(double[], int)} is ignored, as it is throughout the {@code slice}
 * package. Only {@code TaxaDistanceAdapter} returns a non-zero transition correction, and since
 * every evaluation here applies two updates the corrections do not compose into a function of
 * the state, so this operator is not correct for that adapter.
 */
public class MultipleTryOperator extends SliceOperator {

    private static final double ADAPTATION_OFFSET = 10.0;
    private static final double ADAPTATION_EXPONENT = 0.6;
    private static final double MIN_SCALE = 1.0E-12;
    private static final double MAX_SCALE = 1.0E12;
    private static final double INITIAL_VARIANCE = 1.0;
    private static final double MIN_VARIANCE = 1.0E-12;

    /** The scaling of Roberts, Gelman and Gilks, used as the starting point of the adaptation. */
    private static final double INITIAL_SCALE_NUMERATOR = 2.38;

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<List<AdapterGenerator>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Integer> numTriesInput = new Input<>("numTries", "", 20);
    public final Input<Double> targetAcceptanceInput = new Input<>("targetAcceptance", "", 0.44);
    public final Input<Integer> numScalesInput = new Input<>("numScales", "", 3);
    public final Input<Double> scaleSpacingInput = new Input<>("scaleSpacing", "", 8.0);
    public final Input<String> balancingFunctionInput = new Input<>("balancingFunction", "", "barker");

    List<Adapter> adapters;
    Tree tree;
    int totalNumMutable;
    int numTries;
    double targetAcceptance;
    int numScales;
    double scaleSpacing;
    BalancingFunction balancingFunction = BalancingFunction.BARKER;

    KeyedRunningVariances variances;
    int[] coordinateAdapterIndices;
    double[] scaleLadder;
    double globalScale;
    long scaleUpdateCount;

    private long proposalCount;
    private long acceptedCount;
    private long[] acceptedPerRung;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();
        this.tree = this.treeInput.get();
        this.numTries = this.numTriesInput.get();
        this.targetAcceptance = this.targetAcceptanceInput.get();
        this.numScales = this.numScalesInput.get();
        this.scaleSpacing = this.scaleSpacingInput.get();
        this.balancingFunction = BalancingFunction.fromName(this.balancingFunctionInput.get());

        for (AdapterGenerator adapterGenerator : this.adapterGeneratorsInput.get()) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        this.totalNumMutable = 0;
        for (Adapter adapter : this.adapters) {
            this.totalNumMutable += adapter.getNumMutable();
        }

        if (this.totalNumMutable == 0) {
            throw new IllegalArgumentException("MultipleTryOperator requires at least one mutable value");
        }
        if (this.numTries < 1) {
            throw new IllegalArgumentException("numTries must be at least one");
        }
        if (!(0.0 < this.targetAcceptance && this.targetAcceptance < 1.0)) {
            throw new IllegalArgumentException("targetAcceptance must lie strictly between zero and one");
        }
        if (this.numScales < 1) {
            throw new IllegalArgumentException("numScales must be at least one");
        }
        if (!(this.scaleSpacing > 1.0) || !Double.isFinite(this.scaleSpacing)) {
            throw new IllegalArgumentException("scaleSpacing must be finite and greater than one");
        }

        this.initAdaptation();
    }

    /** Sets up the adaptive state. Call after {@code totalNumMutable} is known. */
    void initAdaptation() {
        this.variances = new KeyedRunningVariances(this.adapters.size());
        this.globalScale = INITIAL_SCALE_NUMERATOR / Math.sqrt(this.totalNumMutable);
        this.scaleUpdateCount = 0;

        // a geometric ladder centred on one, so the adapted scale keeps its meaning and a single
        // rung reproduces the plain kernel exactly

        this.scaleLadder = new double[this.numScales];
        for (int i = 0; i < this.numScales; i++) {
            this.scaleLadder[i] = Math.pow(this.scaleSpacing, i - (this.numScales - 1) / 2.0);
        }

        this.proposalCount = 0;
        this.acceptedCount = 0;
        this.acceptedPerRung = new long[this.numScales];

        // which adapter each coordinate belongs to, so a cold key can borrow that adapter's
        // pooled spread; adapter dimensions never change, so this is fixed for the whole run

        this.coordinateAdapterIndices = new int[this.totalNumMutable];

        int idx = 0;
        for (int adapterIndex = 0; adapterIndex < this.adapters.size(); adapterIndex++) {
            int numMutable = this.adapters.get(adapterIndex).getNumMutable();

            for (int i = 0; i < numMutable; i++) {
                this.coordinateAdapterIndices[idx++] = adapterIndex;
            }
        }
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood, State state) {
        // refresh once, before anything reads the mutable vector: CubeAdapter.refresh() redraws
        // the cube ordering, which changes what every coordinate means
        this.refreshAdapters();

        final int nodeId = this.chooseNodeId();
        final double[] currentState = this.getMutable(nodeId);

        // read after the refresh and held for the whole iteration, so the scales stay frozen and
        // the proposal stays symmetric
        final long[] keys = this.getCoordinateKeys(nodeId);

        this.variances.record(keys, this.coordinateAdapterIndices, currentState);
        final double[] scales = this.getProposalScales(keys);

        // SliceMCMC compares posteriors in state space, so the raw value is what the returned
        // Hastings ratio has to cancel against; the weights below use the mutable-space density

        final double currentLogLikelihood = computeCurrentLogLikelihood.get();
        final double currentLogDensity = currentLogLikelihood - this.sumLogJacobianCorrection(nodeId);

        // draw the candidates and weigh each one against the current point

        final double[][] candidates = new double[this.numTries][];
        final int[] rungIndices = new int[this.numTries];
        final double[] logForwardWeights = new double[this.numTries];

        for (int j = 0; j < this.numTries; j++) {
            // drawn per candidate, so the candidates stay independent draws from one mixture kernel
            rungIndices[j] = Randomizer.nextInt(this.scaleLadder.length);
            candidates[j] = this.perturb(currentState, scales, rungIndices[j]);

            Evaluation evaluation = this.evaluateAt(candidates[j], currentState, nodeId, computeCurrentLogLikelihood);
            logForwardWeights[j] = this.logBalancingWeight(evaluation.logDensity(), currentLogDensity);
        }

        final double logForwardTotal = logSumExp(logForwardWeights);
        if (logForwardTotal == Double.NEGATIVE_INFINITY) {
            // every candidate is impossible, so there is nothing to select
            return Double.NEGATIVE_INFINITY;
        }

        final int selectedIndex = sampleIndex(logForwardWeights, logForwardTotal);
        final double[] selected = candidates[selectedIndex];

        // sample the reverse neighbourhood around the selection; its last member is the current
        // point itself, which enters exactly rather than being redrawn

        final double[] reverseLogDensities = new double[this.numTries];

        for (int i = 0; i < this.numTries - 1; i++) {
            double[] reversePoint = this.perturb(selected, scales, Randomizer.nextInt(this.scaleLadder.length));
            reverseLogDensities[i] = this.evaluateAt(reversePoint, currentState, nodeId, computeCurrentLogLikelihood).logDensity();
        }
        reverseLogDensities[this.numTries - 1] = currentLogDensity;

        // move the state onto the selection, which has to happen after the reverse neighbourhood
        // so that the proposal ends with the state where it belongs

        final Evaluation selectedEvaluation = this.evaluateAt(selected, currentState, nodeId, computeCurrentLogLikelihood);
        if (selectedEvaluation.logLikelihood() == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        final double[] logReverseWeights = new double[this.numTries];
        for (int i = 0; i < this.numTries; i++) {
            logReverseWeights[i] = this.logBalancingWeight(reverseLogDensities[i], selectedEvaluation.logDensity());
        }

        final double logAlpha = logForwardTotal - logSumExp(logReverseWeights);

        this.updateGlobalScale(logAlpha);
        this.proposalCount++;

        if (logAlpha < 0.0 && logAlpha <= Math.log(Randomizer.nextDouble())) {
            return Double.NEGATIVE_INFINITY;
        }

        this.acceptedCount++;
        this.acceptedPerRung[rungIndices[selectedIndex]]++;

        // the acceptance decision has already been made here, so cancel out the ratio that
        // SliceMCMC recomputes and let it accept the state we just built
        return currentLogLikelihood - selectedEvaluation.logLikelihood();
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }

        return stateNodes;
    }

    /** The raw posterior and the corresponding density in mutable coordinates at one point. */
    record Evaluation(double logLikelihood, double logDensity) {

        private static final Evaluation IMPOSSIBLE =
                new Evaluation(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
    }

    /**
     * Moves the state onto the given point and evaluates the target there.
     *
     * <p>The base point is reapplied first because adapter updates do not compose:
     * {@code CubeAdapter.update} reads the node heights off the live tree, so applying a
     * candidate on top of a previous candidate does not land where it should.
     */
    Evaluation evaluateAt(double[] mutable, double[] basePoint, int nodeId,
                          Supplier<Double> computeCurrentLogLikelihood) {
        try {
            this.updateAdapters(basePoint, nodeId);
            this.updateAdapters(mutable, nodeId);
        } catch (RuntimeException e) {
            return Evaluation.IMPOSSIBLE;
        }

        double logLikelihood = computeCurrentLogLikelihood.get();
        if (Double.isNaN(logLikelihood) || logLikelihood == Double.NEGATIVE_INFINITY) {
            return Evaluation.IMPOSSIBLE;
        }

        double logDensity = logLikelihood - this.sumLogJacobianCorrection(nodeId);
        if (Double.isNaN(logDensity)) {
            return Evaluation.IMPOSSIBLE;
        }

        return new Evaluation(logLikelihood, logDensity);
    }

    /**
     * The weight {@code g(pi(y) / pi(x))} in logs. For a symmetric proposal these are exactly the
     * admissible multiple-try weights: requiring the Liu-Liang-Wong lambda to be symmetric reduces
     * to {@code g(t) = t g(1/t)}, which is the locally balanced condition, and every member of
     * {@link BalancingFunction} satisfies it.
     */
    double logBalancingWeight(double logDensity, double referenceLogDensity) {
        if (logDensity == Double.NEGATIVE_INFINITY) {
            // an impossible candidate is never selected; MAX in particular needs this, since
            // max(0, -inf) would otherwise hand it a weight of one
            return Double.NEGATIVE_INFINITY;
        }

        if (referenceLogDensity == Double.NEGATIVE_INFINITY) {
            // nothing to weigh against, so every reachable candidate counts the same; the
            // unbounded functions would return positive infinity here and turn the acceptance
            // ratio into a NaN
            return 0.0;
        }

        double logRatio = logDensity - referenceLogDensity;

        return switch (this.balancingFunction) {
            case BARKER -> logDensity - logAddExp(referenceLogDensity, logDensity);
            case SQRT -> 0.5 * logRatio;
            case MIN -> Math.min(0.0, logRatio);
            case MAX -> Math.max(0.0, logRatio);
        };
    }

    /**
     * The locally balanced family, every member satisfying {@code g(t) = t g(1/t)}. BARKER and MIN
     * are bounded by one, so no single candidate can dominate the selection; SQRT and MAX are not.
     */
    enum BalancingFunction {

        /** {@code g(t) = t / (1 + t)}. */
        BARKER,
        /** {@code g(t) = sqrt(t)}. */
        SQRT,
        /** {@code g(t) = min(1, t)}. */
        MIN,
        /** {@code g(t) = max(1, t)}. */
        MAX;

        private static BalancingFunction fromName(String name) {
            try {
                return BalancingFunction.valueOf(name.toUpperCase());
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException("Unknown balancingFunction '" + name, exception);
            }
        }
    }

    @Override
    public String getPerformanceSuggestion() {
        if (this.proposalCount == 0) {
            return "no completed proposals yet";
        }

        StringBuilder suggestion = new StringBuilder();
        suggestion.append(String.format(
                "acceptance %.3f (target %.3f), globalScale %.4g, balancingFunction %s, accepted by rung:",
                (double) this.acceptedCount / this.proposalCount,
                this.targetAcceptance,
                this.globalScale,
                this.balancingFunction.name().toLowerCase()));

        // if the coarsest rung never wins, the ladder is not reaching anything and wants widening
        for (int i = 0; i < this.scaleLadder.length; i++) {
            suggestion.append(String.format(" %.3gx=%.3f",
                    this.scaleLadder[i], (double) this.acceptedPerRung[i] / this.acceptedCount));
        }

        return suggestion.toString();
    }

    private double[] perturb(double[] centre, double[] scales, int rungIndex) {
        double rung = this.scaleLadder[rungIndex];
        double[] perturbed = new double[this.totalNumMutable];

        for (int i = 0; i < this.totalNumMutable; i++) {
            perturbed[i] = centre[i] + rung * scales[i] * Randomizer.nextGaussian();
        }

        return perturbed;
    }

    /**
     * The per-coordinate proposal standard deviations, i.e. the learned marginal standard
     * deviations times the globally adapted scale.
     */
    private double[] getProposalScales(long[] keys) {
        double[] scales = this.variances.getStandardDeviations(keys, this.coordinateAdapterIndices);

        for (int i = 0; i < scales.length; i++) {
            scales[i] *= this.globalScale;
        }

        return scales;
    }

    /**
     * Nudges the global scale towards the target acceptance rate. Driven by the acceptance
     * probability rather than the accept/reject indicator, which has a lower variance.
     *
     * <p>This is deliberately not done in {@code optimize(double)}: the Hastings ratio returned
     * by {@link #proposal} forces the driver's own log alpha to zero, so it carries no signal.
     */
    private void updateGlobalScale(double logAlpha) {
        this.scaleUpdateCount++;

        double acceptanceProbability = Math.exp(Math.min(logAlpha, 0.0));
        double learningRate = Math.pow(this.scaleUpdateCount + ADAPTATION_OFFSET, -ADAPTATION_EXPONENT);
        double logScale = Math.log(this.globalScale)
                + learningRate * (acceptanceProbability - this.targetAcceptance);
        double scale = Math.exp(logScale);

        if (Double.isFinite(scale)) {
            this.globalScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
        }
    }

    private int chooseNodeId() {
        if (this.tree == null) {
            return 0;
        }

        int nodeId = Randomizer.nextInt(this.tree.getNodeCount());

        while (this.tree.getNode(nodeId).isLeaf() || this.tree.getNode(nodeId).isRoot()) {
            nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        }

        return nodeId;
    }

    /**
     * The key of every coordinate: the adapter's own key for it, namespaced by the adapter's
     * position in the list so that two adapters cannot collide.
     */
    private long[] getCoordinateKeys(int nodeId) {
        long[] keys = new long[this.totalNumMutable];

        int idx = 0;
        for (int adapterIndex = 0; adapterIndex < this.adapters.size(); adapterIndex++) {
            Adapter adapter = this.adapters.get(adapterIndex);

            int numMutable = adapter.getNumMutable();
            if (numMutable == 0) continue;

            int[] adapterKeys = adapter.getMutableKeys(nodeId);
            if (adapterKeys.length != numMutable) {
                throw new IllegalStateException(adapter.getClass().getSimpleName() + " returned "
                        + adapterKeys.length + " coordinate keys for " + numMutable + " mutable values");
            }

            for (int i = 0; i < numMutable; i++) {
                keys[idx++] = ((long) adapterIndex << 32) | Integer.toUnsignedLong(adapterKeys[i]);
            }
        }

        return keys;
    }

    private double[] getMutable(int nodeId) {
        double[] mutable = new double[this.totalNumMutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable(nodeId);
            System.arraycopy(adapterMutable, 0, mutable, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();
        }

        return mutable;
    }

    private double sumLogJacobianCorrection(int nodeId) {
        double logCorrection = 0.0;

        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;
            logCorrection += adapter.getLogJacobianCorrection(nodeId);
        }

        return logCorrection;
    }

    private void refreshAdapters() {
        for (Adapter adapter : this.adapters) {
            adapter.refresh();
        }
    }

    private void updateAdapters(double[] mutable, int nodeId) {
        int idx = 0;

        for (Adapter adapter : this.adapters) {
            int numMutable = adapter.getNumMutable();
            if (numMutable == 0) continue;

            double[] adapterMutable = new double[numMutable];
            System.arraycopy(mutable, idx, adapterMutable, 0, numMutable);
            adapter.update(adapterMutable, nodeId);

            idx += numMutable;
        }
    }

    /** Samples an index proportionally to the given log weights. */
    private static int sampleIndex(double[] logWeights, double logTotal) {
        double threshold = Randomizer.nextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < logWeights.length; i++) {
            cumulative += Math.exp(logWeights[i] - logTotal);
            if (threshold < cumulative) return i;
        }

        return logWeights.length - 1;
    }

    private static double logAddExp(double a, double b) {
        double max = Math.max(a, b);
        if (max == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;

        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    private static double logSumExp(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (value > max) max = value;
        }

        if (max == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;

        double sum = 0.0;
        for (double value : values) {
            sum += Math.exp(value - max);
        }

        return max + Math.log(sum);
    }

    /**
     * Welford running mean and variance per coordinate key, alongside one pooled estimate per
     * adapter that stands in for keys not yet seen often enough.
     */
    static final class KeyedRunningVariances {

        private final Map<Long, Welford> perKey = new HashMap<>();
        private final Welford[] perAdapter;

        KeyedRunningVariances(int numAdapters) {
            this.perAdapter = new Welford[numAdapters];

            for (int i = 0; i < numAdapters; i++) {
                this.perAdapter[i] = new Welford();
            }
        }

        void record(long[] keys, int[] adapterIndices, double[] values) {
            for (double value : values) {
                // a non-finite coordinate would poison the estimates for good
                if (!Double.isFinite(value)) return;
            }

            for (int i = 0; i < values.length; i++) {
                this.perKey.computeIfAbsent(keys[i], key -> new Welford()).record(values[i]);
                this.perAdapter[adapterIndices[i]].record(values[i]);
            }
        }

        double[] getStandardDeviations(long[] keys, int[] adapterIndices) {
            double[] standardDeviations = new double[keys.length];

            for (int i = 0; i < keys.length; i++) {
                Welford welford = this.perKey.get(keys[i]);

                // a key seen once or not at all borrows its adapter's pooled spread, which beats
                // starting it off at the arbitrary default
                double variance = welford == null || welford.count < 2
                        ? this.perAdapter[adapterIndices[i]].getVariance()
                        : welford.getVariance();

                standardDeviations[i] = Math.sqrt(variance);
            }

            return standardDeviations;
        }
    }

    private static final class Welford {

        private long count;
        private double mean;
        private double sumSquares;

        private void record(double value) {
            this.count++;

            double delta = value - this.mean;
            this.mean += delta / this.count;
            this.sumSquares += delta * (value - this.mean);
        }

        private double getVariance() {
            if (this.count < 2) return INITIAL_VARIANCE;

            return Math.max(this.sumSquares / (this.count - 1), MIN_VARIANCE);
        }
    }

    @Override
    public double getCoercableParameterValue() {
        return globalScale;
    }
}
