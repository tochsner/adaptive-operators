package weightoptimization;

import adapters.Adapter;
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

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Operator that selects among a list of operators and re-weights them once, based on the effective
 * sample size per second each of them achieves on its own proxy trace.
 *
 * <p>Every operator is paired with its own {@link Adapter}, which should return the quantities that
 * operator is meant to mix (for instance a topology summary for a tree operator). The ESS of an
 * adapter's coordinates is therefore attributed to its operator alone.
 *
 * <p>The run goes through three phases:
 * <ol>
 *   <li>warm-up: the traces of all adapters are recorded until the smallest ESS over all of them
 *   exceeds {@code lowThreshold};</li>
 *   <li>monitoring: the traces keep being recorded, along with the time spent in each operator,
 *   until the smallest ESS over all of them exceeds {@code highThreshold};</li>
 *   <li>adapted: the weights computed from the monitoring are used for the rest of the run.</li>
 * </ol>
 *
 * <p>The chain is as good as its worst trace, so the weights aim at equal ESS per second on all
 * traces: an operator's new selection probability is proportional to the number of applications it
 * needed per unit of ESS its adapter's trace gained between the start and the end of the monitoring
 * (smallest over its coordinates), mixed with a floor so that no operator is switched off entirely.
 *
 * <p>The ESS of a trace is computed by a {@link ThinnedEssTracker}, which leaves out the first 10%
 * of the trace as burn-in, like Tracer does.
 *
 * <p>Application times are not measured here; they are supplied through
 * {@link #recordApplicationTime(long)}, which {@link mcmc.OperatorTimingMCMC} calls with the
 * duration of the whole step (proposal and recomputation of the posterior).
 */
public class ESSWeightAdaptationOperator extends Operator {

    final public Input<List<Operator>> operatorsInput = new Input<>(
            "operator", "Operators to select from. Their weights give the initial selection probabilities.", new ArrayList<>()
    );
    final public Input<List<Adapter>> adaptersInput = new Input<>(
            "adapter", "One adapter per operator, in the same order, whose immutable values are the scalar traces the ESS of that operator is computed on.",
            new ArrayList<>()
    );
    final public Input<Double> lowThresholdInput = new Input<>(
            "lowThreshold", "Smallest ESS above which monitoring starts.", 25.0
    );
    final public Input<Double> highThresholdInput = new Input<>(
            "highThreshold", "The weights are adjusted once the smallest ESS exceeds highThreshold.", 75.0
    );
    final public Input<Double> minWeightFractionInput = new Input<>(
            "minWeightFraction", "Selection probability every operator keeps after adaptation.", 0.1
    );

    private static final int SAMPLING_INTERVAL = 100;
    private static final double MIN_ESS_GAIN = 1.0;

    private enum Phase {WARM_UP, MONITORING, ADAPTED}

    private List<Operator> operators;
    private List<Adapter> adapters;
    private double[] probabilities;
    private double[] initialProbabilities;

    private Phase phase = Phase.WARM_UP;
    private int lastOperatorIdx;
    private double[][] valuesBeforeProposal;
    private int numSteps;
    private boolean isRecordedStep;

    // indexed by operator, then by coordinate of that operator's adapter
    private List<List<ThinnedEssTracker>> trackers;
    private int numSamples;

    private long[] nanosPerOperator;
    private int[] numTimedApplications;
    private double[] warmUpEss;
    private double[] endEss;
    private int[] numAccepted;
    private int[] numRejected;

    @Override
    public void initAndValidate() {
        this.operators = this.operatorsInput.get();
        this.adapters = this.adaptersInput.get();

        this.checkInputs();

        this.probabilities = this.computeInitialProbabilities();
        this.initialProbabilities = this.probabilities.clone();

        this.trackers = this.createTrackers();
        this.nanosPerOperator = new long[this.operators.size()];
        this.numTimedApplications = new int[this.operators.size()];
        this.numAccepted = new int[this.operators.size()];
        this.numRejected = new int[this.operators.size()];
    }

    private void checkInputs() {
        if (this.operators.isEmpty()) {
            throw new IllegalArgumentException("At least one operator is required.");
        }
        if (this.adapters.size() != this.operators.size()) {
            throw new IllegalArgumentException("Exactly one adapter per operator is required.");
        }
        if (this.lowThresholdInput.get() >= this.highThresholdInput.get()) {
            throw new IllegalArgumentException("lowThreshold must be smaller than highThreshold.");
        }
        if (this.minWeightFractionInput.get() * this.operators.size() >= 1.0) {
            throw new IllegalArgumentException("minWeightFraction times the number of operators must be below one.");
        }
    }

    private double[] computeInitialProbabilities() {
        double totalWeight = this.operators.stream().mapToDouble(Operator::getWeight).sum();

        return this.operators.stream().mapToDouble(operator -> operator.getWeight() / totalWeight).toArray();
    }

    private List<List<ThinnedEssTracker>> createTrackers() {
        List<List<ThinnedEssTracker>> newTrackers = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            List<ThinnedEssTracker> operatorTrackers = new ArrayList<>();

            for (int k = 0; k < adapter.getNumImmutable(); k++) {
                operatorTrackers.add(new ThinnedEssTracker(1));
            }

            newTrackers.add(operatorTrackers);
        }

        return newTrackers;
    }

    @Override
    public double proposal() {
        // only every few steps are recorded, which also saves reading the adapters on the others
        this.numSteps++;
        this.isRecordedStep = this.phase != Phase.ADAPTED && this.numSteps % SAMPLING_INTERVAL == 0;

        if (this.isRecordedStep) {
            this.valuesBeforeProposal = this.readAdapters();
        }

        this.lastOperatorIdx = Randomizer.binarySearchSampling(this.computeCumulativeProbabilities());

        return this.operators.get(this.lastOperatorIdx).proposal();
    }

    private double[][] readAdapters() {
        double[][] values = new double[this.adapters.size()][];

        for (int i = 0; i < values.length; i++) {
            values[i] = this.adapters.get(i).getImmutable(0);
        }

        return values;
    }

    private double[] computeCumulativeProbabilities() {
        double[] cumulative = new double[this.probabilities.length];

        double sum = 0.0;
        for (int i = 0; i < cumulative.length; i++) {
            sum += this.probabilities[i];
            cumulative[i] = sum;
        }

        return cumulative;
    }

    @Override
    public void accept() {
        super.accept();
        this.operators.get(this.lastOperatorIdx).accept();
        this.numAccepted[this.lastOperatorIdx]++;

        // the state still holds the accepted proposal at this point
        if (this.isRecordedStep) {
            this.recordSample(this.readAdapters());
        }
    }

    @Override
    public void reject(int reason) {
        super.reject(reason);
        this.operators.get(this.lastOperatorIdx).reject(reason);
        this.numRejected[this.lastOperatorIdx]++;

        // the state has not been restored yet, so the value from before the proposal is the current one
        if (this.isRecordedStep) {
            this.recordSample(this.valuesBeforeProposal);
        }
    }

    private void recordSample(double[][] values) {
        if (this.phase == Phase.ADAPTED) return;

        for (int i = 0; i < values.length; i++) {
            for (int k = 0; k < values[i].length; k++) {
                this.trackers.get(i).get(k).record(values[i][k]);
            }
        }
        this.numSamples++;

        if (this.phase == Phase.WARM_UP) {
            this.updateWarmUp();
        } else {
            this.updateMonitoring();
        }
    }

    private void updateWarmUp() {
        this.warmUpEss = this.computeOperatorEss();

        if (Arrays.stream(this.warmUpEss).min().orElse(0.0) > this.lowThresholdInput.get()) {
            this.phase = Phase.MONITORING;

            Log.info.println("Monitoring operator behavior started. Current ESS: " + Arrays.toString(this.warmUpEss));
        }
    }

    private void updateMonitoring() {
        this.endEss = this.computeOperatorEss();

        if (Arrays.stream(this.endEss).min().orElse(0.0) > this.highThresholdInput.get()) {
            this.adaptWeights();
            this.phase = Phase.ADAPTED;
            this.trackers = null;
        }
    }

    /**
     * Smallest ESS over the coordinates of each operator's adapter. An adapter without coordinates
     * gets infinite ESS, so that it cannot hold the others back.
     */
    private double[] computeOperatorEss() {
        return this.trackers.stream()
                .mapToDouble(operatorTrackers -> operatorTrackers.stream()
                        .mapToDouble(ThinnedEssTracker::getEss)
                        .min()
                        .orElse(Double.POSITIVE_INFINITY))
                .toArray();
    }

    private void adaptWeights() {
        double[] needs = this.computeNeeds();
        double totalNeed = Arrays.stream(needs).sum();

        if (!(totalNeed > 0.0) || Double.isInfinite(totalNeed)) {
            Log.warning.println("ESSWeightAdaptationOperator: no operator was applied in the monitoring window, keeping the initial weights.");
            return;
        }

        int numOperators = this.operators.size();
        double floor = this.minWeightFractionInput.get();
        for (int i = 0; i < numOperators; i++) {
            this.probabilities[i] = floor + (1.0 - numOperators * floor) * needs[i] / totalNeed;
        }

        Log.info.println("Adapted operator weights after " + this.numSamples + " samples:");
        Log.info.println(this.formatWeightTable());
    }

    /**
     * How many applications each operator needed per unit of ESS gained on its own trace.
     *
     * <p>The gain is the difference between the ESS at the end and at the start of the monitoring,
     * so that the ESS the trace had already reached during warm-up does not count as mixing.
     *
     * <p>The chain is as good as its worst trace, so the time is best spent equalising the ESS per
     * second of all traces. An operator whose trace is already mixing well therefore gets less
     * weight, and one whose trace gains little per application gets more. A gain below one effective
     * sample is counted as one, so an operator that does not move its trace at all cannot take over.
     */
    private double[] computeNeeds() {
        double[] needs = new double[this.operators.size()];

        for (int i = 0; i < needs.length; i++) {
            if (!Double.isFinite(this.endEss[i])) continue;

            double gain = this.endEss[i] - this.warmUpEss[i];
            needs[i] = this.numTimedApplications[i] / Math.max(gain, MIN_ESS_GAIN);
        }

        return needs;
    }

    /**
     * Formats the selection probability, acceptance count and acceptance rate of every wrapped
     * operator as a table, alongside the probability it started with, the ESS of its adapter at the
     * start of the monitoring, the most recently evaluated ESS of its adapter during it (which the
     * weights are based on) and its average time per application in microseconds.
     */
    public String formatWeightTable() {
        int nameWidth = this.operators.stream().mapToInt(operator -> operator.getName().length()).max().orElse(0);
        nameWidth = Math.max(nameWidth, "Operator".length());

        String rowFormat = "%-" + nameWidth + "s %10s %10s %10s %10s %10s %10s %10s %10s%n";

        StringBuilder builder = new StringBuilder();
        builder.append(String.format(rowFormat, "Operator", "Pr(m) init", "Pr(m)", "#accept", "#reject", "Pr(acc|m)", "ESS start", "ESS end", "time [us]"));

        for (int i = 0; i < this.operators.size(); i++) {
            int numProposals = this.numAccepted[i] + this.numRejected[i];
            String acceptanceRate = numProposals > 0
                    ? String.format("%.5f", (double) this.numAccepted[i] / numProposals)
                    : "-";

            builder.append(String.format(
                    rowFormat, this.operators.get(i).getName(),
                    String.format("%.5f", this.initialProbabilities[i]),
                    String.format("%.5f", this.probabilities[i]),
                    this.numAccepted[i], this.numRejected[i], acceptanceRate,
                    this.warmUpEss == null ? "-" : String.format("%.1f", this.warmUpEss[i]),
                    this.endEss == null ? "-" : String.format("%.1f", this.endEss[i]),
                    this.numTimedApplications[i] == 0
                            ? "-"
                            : String.format("%.1f", this.nanosPerOperator[i] * 1e-3 / this.numTimedApplications[i])
            ));
        }

        return builder.toString();
    }

    @Override
    public void optimize(double logAlpha) {
        this.operators.get(this.lastOperatorIdx).optimize(logAlpha);
    }

    /**
     * Records how long the last step took, attributing it to the operator that was selected for it.
     */
    public void recordApplicationTime(long nanos) {
        if (this.phase == Phase.MONITORING) {
            this.nanosPerOperator[this.lastOperatorIdx] += nanos;
            this.numTimedApplications[this.lastOperatorIdx]++;
        }
    }

    /**
     * Whether the weights have been adjusted yet, as opposed to still being the initial ones.
     */
    public boolean isAdapted() {
        return this.phase == Phase.ADAPTED;
    }

    @Override
    public void setOperatorSchedule(OperatorSchedule operatorSchedule) {
        super.setOperatorSchedule(operatorSchedule);

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
            JSONObject json = readStoredJson(super::storeToFile);

            JSONArray innerOperators = new JSONArray();
            for (Operator operator : this.operators) {
                innerOperators.put(readStoredJson(operator::storeToFile));
            }
            json.put("operators", innerOperators);

            out.print(json);
        } catch (JSONException e) {
            // failed to log the operator in the state file, report and continue
            e.printStackTrace();
        }
    }

    static JSONObject readStoredJson(Consumer<PrintWriter> store) throws JSONException {
        StringWriter buffer = new StringWriter();
        PrintWriter writer = new PrintWriter(buffer);

        store.accept(writer);
        writer.flush();

        return new JSONObject(buffer.toString());
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
