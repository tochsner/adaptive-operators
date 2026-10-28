package weightoptimization;

import adapters.Adapter;
import beast.base.core.BEASTObjectStore;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
import beast.base.inference.StateNode;
import beast.base.internal.json.JSONArray;
import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;
import beast.base.util.Randomizer;
import mcmc.ThinnedEssTracker;
import schedule.ESSWeightOperatorSchedule;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Operator that wraps a group of operators and re-weights the group as a whole once, based on how
 * much ESS the group gains on its own adapter's trace while a reference trace gains a fixed amount
 * of ESS.
 *
 * <p>Within the group, an operator is selected with probability proportional to its own weight.
 * These relative weights never change; only the weight of the group, which is the weight of this
 * operator in the schedule, is adapted.
 *
 * <p>The run goes through three phases:
 * <ol>
 *   <li>warm-up: the traces are recorded until the ESS of the reference adapter exceeds
 *   {@code lowThreshold};</li>
 *   <li>monitoring: the traces keep being recorded until the ESS of the reference adapter exceeds
 *   {@code highThreshold};</li>
 *   <li>adapted: the weight computed from the monitoring is used for the rest of the run.</li>
 * </ol>
 *
 * <p>The reference trace gains ESS at the pace of the chain as a whole, so it tells how much mixing
 * is going on elsewhere in the same stretch of the run. The weight of this operator is multiplied by
 * the ratio of the ESS the reference gained to the ESS the group's adapter gained between the start
 * and the end of the monitoring: a group that mixes its own trace slower than the reference is
 * selected more often, and a faster one less often. The factor is limited to {@code maxWeightFactor} in either direction.
 *
 * <p>The traces are sampled whenever this operator is selected, so ESS is counted per sample taken
 * that way. This is consistent between the two traces, which is all the ratio relies on. The ESS of
 * a trace is computed by a {@link ThinnedEssTracker}, which leaves out the first 10% of the trace as
 * burn-in, like Tracer does.
 *
 * <p>The schedule is asked to recompute its selection probabilities if it supports that, see
 * {@link ESSWeightOperatorSchedule#reweight()}.
 */
public class GroupESSWeightAdaptationOperator extends Operator {

    final public Input<List<Operator>> operatorsInput = new Input<>(
            "operator", "Operators of the group. Their weights give the fixed selection probabilities within the group.", new ArrayList<>()
    );
    final public Input<Adapter> operatorAdapterInput = new Input<>(
            "operatorAdapter", "Adapter whose immutable values are the scalar traces the ESS gain of the group is computed on.", Input.Validate.REQUIRED
    );
    final public Input<Adapter> referenceAdapterInput = new Input<>(
            "referenceAdapter", "Adapter whose immutable values are the scalar traces that decide when to start and stop monitoring.", Input.Validate.REQUIRED
    );
    final public Input<Double> lowThresholdInput = new Input<>(
            "lowThreshold", "ESS of the reference above which monitoring starts.", 25.0
    );
    final public Input<Double> highThresholdInput = new Input<>(
            "highThreshold", "ESS of the reference above which monitoring stops.", 75.0
    );
    final public Input<Double> maxWeightFactorInput = new Input<>(
            "maxWeightFactor", "The weight is multiplied or divided by at most this factor.", 10.0
    );

    private static final int SAMPLING_INTERVAL = 10;
    private static final double MIN_ESS_GAIN = 1.0;

    private enum Phase {WARM_UP, MONITORING, ADAPTED}

    private List<Operator> operators;
    private double[] cumulativeProbabilities;
    private Adapter operatorAdapter;
    private Adapter referenceAdapter;
    private OperatorSchedule schedule;

    private Phase phase = Phase.WARM_UP;
    private double initialWeight;
    private int lastOperatorIdx;
    private int numApplications;
    private boolean isRecordedStep;
    private double[] operatorValuesBeforeProposal;
    private double[] referenceValuesBeforeProposal;

    // indexed by coordinate of the respective adapter
    private List<ThinnedEssTracker> operatorTrackers;
    private List<ThinnedEssTracker> referenceTrackers;
    private int numSamples;

    // most recently evaluated ESS values, NaN until first evaluated
    private double referenceWarmUpEss = Double.NaN;
    private double operatorWarmUpEss = Double.NaN;
    private double referenceEndEss = Double.NaN;
    private double operatorEndEss = Double.NaN;

    @Override
    public void initAndValidate() {
        this.operators = this.operatorsInput.get();
        this.operatorAdapter = this.operatorAdapterInput.get();
        this.referenceAdapter = this.referenceAdapterInput.get();

        this.checkInputs();

        this.cumulativeProbabilities = this.computeCumulativeProbabilities();
        this.initialWeight = this.getWeight();
        this.operatorTrackers = this.createTrackers(this.operatorAdapter);
        this.referenceTrackers = this.createTrackers(this.referenceAdapter);
    }

    private void checkInputs() {
        if (this.operators.isEmpty()) {
            throw new IllegalArgumentException("At least one operator is required.");
        }
        if (this.referenceAdapter.getNumImmutable() == 0) {
            throw new IllegalArgumentException("The reference adapter needs at least one immutable value.");
        }
        if (this.lowThresholdInput.get() >= this.highThresholdInput.get()) {
            throw new IllegalArgumentException("lowThreshold must be smaller than highThreshold.");
        }
        if (this.maxWeightFactorInput.get() < 1.0) {
            throw new IllegalArgumentException("maxWeightFactor must be at least one.");
        }
    }

    private double[] computeCumulativeProbabilities() {
        double totalWeight = this.operators.stream().mapToDouble(Operator::getWeight).sum();
        double[] cumulative = new double[this.operators.size()];

        double sum = 0.0;
        for (int i = 0; i < cumulative.length; i++) {
            sum += this.operators.get(i).getWeight() / totalWeight;
            cumulative[i] = sum;
        }

        return cumulative;
    }

    private List<ThinnedEssTracker> createTrackers(Adapter adapter) {
        List<ThinnedEssTracker> trackers = new ArrayList<>();

        for (int k = 0; k < adapter.getNumImmutable(); k++) {
            trackers.add(new ThinnedEssTracker(1));
        }

        return trackers;
    }

    @Override
    public double proposal() {
        // only every few applications are recorded, which also saves reading the adapters on the others
        this.numApplications++;
        this.isRecordedStep = this.phase != Phase.ADAPTED && this.numApplications % SAMPLING_INTERVAL == 0;

        if (this.isRecordedStep) {
            this.operatorValuesBeforeProposal = this.operatorAdapter.getImmutable(0);
            this.referenceValuesBeforeProposal = this.referenceAdapter.getImmutable(0);
        }

        this.lastOperatorIdx = Randomizer.binarySearchSampling(this.cumulativeProbabilities);

        return this.operators.get(this.lastOperatorIdx).proposal();
    }

    @Override
    public void accept() {
        super.accept();
        this.operators.get(this.lastOperatorIdx).accept();

        // the state still holds the accepted proposal at this point
        if (this.isRecordedStep) {
            this.recordSample(this.operatorAdapter.getImmutable(0), this.referenceAdapter.getImmutable(0));
        }
    }

    @Override
    public void reject(int reason) {
        super.reject(reason);
        this.operators.get(this.lastOperatorIdx).reject(reason);

        // the state has not been restored yet, so the value from before the proposal is the current one
        if (this.isRecordedStep) {
            this.recordSample(this.operatorValuesBeforeProposal, this.referenceValuesBeforeProposal);
        }
    }

    private void recordSample(double[] operatorValues, double[] referenceValues) {
        this.recordValues(this.operatorTrackers, operatorValues);
        this.recordValues(this.referenceTrackers, referenceValues);
        this.numSamples++;

        if (this.phase == Phase.WARM_UP) {
            this.updateWarmUp();
        } else {
            this.updateMonitoring();
        }
    }

    private void recordValues(List<ThinnedEssTracker> trackers, double[] values) {
        for (int k = 0; k < values.length; k++) {
            trackers.get(k).record(values[k]);
        }
    }

    private void updateWarmUp() {
        this.referenceWarmUpEss = computeMinEss(this.referenceTrackers);
        this.operatorWarmUpEss = computeMinEss(this.operatorTrackers);

        if (this.referenceWarmUpEss > this.lowThresholdInput.get()) {
            this.phase = Phase.MONITORING;

            Log.info.println("Monitoring " + this.getID() + " started. Current reference ESS: " + this.referenceWarmUpEss);
        }
    }

    private void updateMonitoring() {
        this.referenceEndEss = computeMinEss(this.referenceTrackers);
        this.operatorEndEss = computeMinEss(this.operatorTrackers);

        if (this.referenceEndEss > this.highThresholdInput.get()) {
            this.adaptWeight();
            this.phase = Phase.ADAPTED;
            this.operatorTrackers = null;
            this.referenceTrackers = null;
        }
    }

    /**
     * Smallest ESS over the coordinates of an adapter. An adapter without coordinates gets infinite
     * ESS.
     */
    private static double computeMinEss(List<ThinnedEssTracker> trackers) {
        return trackers.stream()
                .mapToDouble(ThinnedEssTracker::getEss)
                .min()
                .orElse(Double.POSITIVE_INFINITY);
    }

    /**
     * Scales the weight by how much more ESS the reference gained than the group's own trace between
     * the start and the end of the monitoring. A gain below one effective sample is counted as one,
     * so a group that does not move its trace at all cannot take over.
     */
    private void adaptWeight() {
        if (!Double.isFinite(this.operatorEndEss)) {
            Log.warning.println("GroupESSWeightAdaptationOperator: the adapter of " + this.getID() + " has no values, keeping the initial weight.");
            return;
        }

        double maxFactor = this.maxWeightFactorInput.get();
        double referenceGain = this.referenceEndEss - this.referenceWarmUpEss;
        double operatorGain = this.operatorEndEss - this.operatorWarmUpEss;
        double factor = referenceGain / Math.max(operatorGain, MIN_ESS_GAIN);
        factor = Math.min(Math.max(factor, 1.0 / maxFactor), maxFactor);

        this.setWeight(this.initialWeight * factor);

        Log.info.println("Adapted weight of " + this.getID() + " after " + this.numSamples + " samples:");
        Log.info.println(this.formatWeightSummary());
    }

    private void setWeight(double weight) {
        this.m_pWeight.setValue(weight, BEASTObjectStore.INSTANCE.getBEASTObject(this.m_pWeight));

        if (this.schedule instanceof ESSWeightOperatorSchedule essSchedule) {
            essSchedule.reweight();
        } else {
            Log.warning.println("GroupESSWeightAdaptationOperator: the operator schedule is not an ESSWeightOperatorSchedule, "
                    + "so the new weight of " + this.getID() + " only takes effect once the schedule recomputes its probabilities.");
        }
    }

    /**
     * Formats the initial and current weight of the group alongside the ESS of the reference and of
     * the group's adapter at the start and at the end of the monitoring (which the weight is based
     * on).
     */
    public String formatWeightSummary() {
        return String.format(
                "%s: weight %.5f -> %.5f, reference ESS start %s end %s, group ESS start %s end %s%n",
                this.getID(), this.initialWeight, this.getWeight(),
                this.phase == Phase.WARM_UP ? "-" : String.format("%.1f", this.referenceWarmUpEss),
                this.phase != Phase.ADAPTED ? "-" : String.format("%.1f", this.referenceEndEss),
                this.phase == Phase.WARM_UP ? "-" : String.format("%.1f", this.operatorWarmUpEss),
                this.phase != Phase.ADAPTED ? "-" : String.format("%.1f", this.operatorEndEss)
        );
    }

    /**
     * Whether the weight has been adjusted yet, as opposed to still being the initial one.
     */
    public boolean isAdapted() {
        return this.phase == Phase.ADAPTED;
    }

    @Override
    public void optimize(double logAlpha) {
        this.operators.get(this.lastOperatorIdx).optimize(logAlpha);
    }

    @Override
    public void setOperatorSchedule(OperatorSchedule operatorSchedule) {
        super.setOperatorSchedule(operatorSchedule);

        this.schedule = operatorSchedule;
        for (Operator operator : this.operators) {
            operator.setOperatorSchedule(operatorSchedule);
        }
    }

    @Override
    public List<StateNode> listStateNodes() {
        Set<StateNode> stateNodes = new HashSet<>();

        for (Operator operator : this.operators) {
            stateNodes.addAll(operator.listStateNodes());
        }

        return new ArrayList<>(stateNodes);
    }

    /**
     * Stores this operator's state together with the states of the wrapped operators, so that their
     * tuning survives a resume. The schedule only knows about this operator, so it cannot store them
     * itself.
     */
    @Override
    public void storeToFile(PrintWriter out) {
        try {
            JSONObject json = ESSWeightAdaptationOperator.readStoredJson(super::storeToFile);

            // json cannot hold NaN, so ESS values that were not evaluated yet are left out
            this.putIfEvaluated(json, "referenceWarmUpEss", this.referenceWarmUpEss);
            this.putIfEvaluated(json, "operatorWarmUpEss", this.operatorWarmUpEss);
            this.putIfEvaluated(json, "referenceEndEss", this.referenceEndEss);
            this.putIfEvaluated(json, "operatorEndEss", this.operatorEndEss);

            JSONArray innerOperators = new JSONArray();
            for (Operator operator : this.operators) {
                innerOperators.put(ESSWeightAdaptationOperator.readStoredJson(operator::storeToFile));
            }
            json.put("operators", innerOperators);

            out.print(json);
        } catch (JSONException e) {
            // failed to log the operator in the state file, report and continue
            e.printStackTrace();
        }
    }

    private void putIfEvaluated(JSONObject json, String key, double ess) throws JSONException {
        if (Double.isFinite(ess)) {
            json.put(key, ess);
        }
    }

    @Override
    public void restoreFromFile(JSONObject o) {
        super.restoreFromFile(o);

        // state files written before the wrapped operators were stored lack them
        if (!o.has("operators")) return;

        try {
            JSONArray innerOperators = o.getJSONArray("operators");

            if (innerOperators.length() != this.operators.size()) {
                throw new IllegalArgumentException("Cannot resume because the state file has " + innerOperators.length()
                        + " operators inside " + this.getID() + " but there are " + this.operators.size() + ".");
            }

            for (int i = 0; i < this.operators.size(); i++) {
                this.operators.get(i).restoreFromFile(innerOperators.getJSONObject(i));
            }
        } catch (JSONException e) {
            // failed to restore from the state file, report and continue
            e.printStackTrace();
        }
    }
}
