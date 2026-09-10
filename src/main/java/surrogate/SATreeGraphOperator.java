package surrogate;

import adapters.Adapter;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Tree;
import beast.base.inference.*;
import beast.base.spec.evolution.tree.ClusterTree;
import beast.base.util.Randomizer;
import mala.SubsetJukesCantorDistance;

import java.util.ArrayList;
import java.util.List;

public class SATreeGraphOperator extends TreeOperator {

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    public final Input<List<Operator>> operatorsInput = new Input<>("operator", "", new ArrayList<>());
    public final Input<State> stateInput = new Input<>("state", "");
    public final Input<Distribution> posteriorInput = new Input<>("posterior", "");

    Tree tree;
    List<Adapter> adapters;
    Alignment alignment;
    List<Operator> operators;
    State state;
    Distribution posterior;

    boolean hasBeenInitialized;
    OperatorSchedule operatorSchedule;
    List<Tree> topologySamples;
    List<SampledState> samples;

    @Override
    public void initAndValidate() {
        this.hasBeenInitialized = false;
        this.tree = this.treeInput.get();
        this.adapters = this.adaptersInput.get();
        this.alignment = this.alignmentInput.get();
        this.operators = this.operatorsInput.get();
        this.state = this.stateInput.get();
        this.posterior = this.posteriorInput.get();
    }

    private void initEverything() {
        this.operatorSchedule = this.buildOperatorSchedule();
        this.topologySamples = this.sampleTopologies();
        this.samples = this.sampleEverything();
        this.hasBeenInitialized = true;
    }

    private OperatorSchedule buildOperatorSchedule() {
        OperatorSchedule operatorSchedule = new OperatorSchedule();
        operatorSchedule.operatorsInput.setTypedValue(this.operators, operatorSchedule);
        operatorSchedule.initAndValidate();
        return operatorSchedule;
    }

    private List<Tree> sampleTopologies() {
        int numTopologies = 10;
        double sitesFraction = 0.5;

        List<Tree> sampledTopologies = new ArrayList<>();
        for (int i = 0; i < numTopologies; i++) {
            ClusterTree tree = new ClusterTree();
            tree.clusterTypeInput.setTypedValue(ClusterTree.Type.upgma, tree);
            tree.distanceInput.setTypedValue(new SubsetJukesCantorDistance(sitesFraction), tree);
            tree.dataInput.setTypedValue(this.alignment, tree);
            tree.m_traitList.setTypedValue(this.tree.m_traitList.get(), tree);
            tree.initAndValidate();

            sampledTopologies.add(tree);
        }

        return sampledTopologies;
    }

    private List<SampledState> sampleEverything() {
        int chainLength = 100_000;
        int burnIn = 80_000;
        double collectionProbability = 0.01;

        // we run simulated annealing

        List<SampledState> sampledStates = new ArrayList<>();

        int ii = 0;
        for (Tree topology : this.topologySamples) {
            // set as initial tree
            System.out.println(ii++);

            // the swap has to happen inside a transaction, otherwise the state is never told the
            // tree changed and the tree likelihood stays off the calculation node path
            this.state.store(-1);
            this.tree.startEditing(this);

            double oldHeight = this.tree.getRoot().getHeight();
            this.tree.assignFrom(topology);

            double newHeight = this.tree.getRoot().getHeight();
            this.tree.scale(oldHeight / newHeight);

            // assignFrom replaces every Node object, so no incremental cache survives it
            this.robustlyRecalculate();

            // run simulated annealing

            for (int i = 0; i < chainLength; i++) {
                // propose
                boolean sample = Randomizer.nextDouble() < collectionProbability && burnIn < i;

                this.proposeSimulatedAnnealingSample(i, sampledStates, sample);
            }
        }

        return sampledStates;
    }

    double newLogLikelihood = Double.NEGATIVE_INFINITY;
    double oldLogLikelihood = Double.NEGATIVE_INFINITY;
    double logAlpha = Double.NEGATIVE_INFINITY;
    private void proposeSimulatedAnnealingSample(long sampleNr, List<SampledState> sampledStates, boolean sample) {
        double temperature = 0.1;

        this.state.store(sampleNr);

        final Operator operator = this.operatorSchedule.selectOperator();
        final double logHastingsRatio = operator.proposal();

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {

            if (operator.requiresStateInitialisation()) {
                this.state.storeCalculationNodes();
                this.state.checkCalculationNodesDirtiness();
            }

            this.newLogLikelihood = this.posterior.calculateLogP();

            // simulated annealing acceptance: scale the log-likelihood difference by the
            // temperature and keep the Hastings ratio as an unscaled proposal correction.
            this.logAlpha = (this.newLogLikelihood - this.oldLogLikelihood) / temperature + logHastingsRatio;

            boolean accept = this.logAlpha >= 0
                    || (this.logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(this.logAlpha));

            if (accept) {
                // accept
                this.oldLogLikelihood = this.newLogLikelihood;
                this.state.acceptCalculationNodes();
                operator.accept();

                if (sample) {
                    // we store the current state

                    List<double[]> mutableStates = new ArrayList<>();
                    List<double[]> immutableStates = new ArrayList<>();

                    for (Adapter adapter : this.adapters) {
                        double[] mutableState = adapter.getMutable(0);
                        double[] immutableState = adapter.getImmutable(0);
                        mutableStates.add(mutableState);
                        immutableStates.add(immutableState);
                    }

                    sampledStates.add(
                            new SampledState(mutableStates, immutableStates, this.newLogLikelihood)
                    );
                }
            } else {
                // reject
                operator.reject(this.newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                this.state.restore();
                this.state.restoreCalculationNodes();
            }
            this.state.setEverythingDirty(false);

        } else {

            this.logAlpha = Double.NEGATIVE_INFINITY;
            // operation failed
            operator.reject(-2);
            this.state.restore();
            if (!operator.requiresStateInitialisation()) {
                this.state.setEverythingDirty(false);
                this.state.restoreCalculationNodes();
            }

        }
    }

    /**
     * Recompute the whole posterior from scratch and leave the cache clean and consistent with the
     * current state. Must only be called after any state.restore(), never before, because
     * setEverythingDirty(true) marks every node dirty without backing it up.
     */
    private void robustlyRecalculate() {
        this.state.setEverythingDirty(true);
        this.state.checkCalculationNodesDirtiness();
        this.oldLogLikelihood = this.posterior.calculateLogP();
        this.state.setEverythingDirty(false);
        this.state.acceptCalculationNodes();
    }

    @Override
    public double proposal() {
        if (!this.hasBeenInitialized) {
            this.initEverything();

            // the inner annealing run left the outer chain's single-buffered cache unusable
            this.robustlyRecalculate();
        }

        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }
}
