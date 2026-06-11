package mala;

import adapters.Adapter;
import adapters.AdapterGenerator;
import adaptiveoperators.AdaptiveOperator;
import adaptiveoperators.CenteredMultivariateNormalSampler;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import ml.MLP;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MALAOperator extends AdaptiveOperator {

    public final Input<List<AdapterGenerator<?>>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());
    public final Input<List<Operator>> alternativeOperatorsInput = new Input<>(
            "alternativeOperator",
            "",
            new ArrayList<>());

    private Tree tree;

    private List<Adapter> adapters;
    private int totalNumMutable;
    private int totalNumImmutable;

    private List<Operator> alternativeOperators;

    private double alpha = 1.0;
    private CenteredMultivariateNormalSampler covarianceSampler;
    private MLP mlp;

    private int BURN_IN = 10_000;
    private int JUST_TRAINING = 20_000;
    private double offset = 0.0;
    private int count = 0;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.adapters = this.adaptersInput.get();
        this.alternativeOperators = this.alternativeOperatorsInput.get();

        for (AdapterGenerator<?> adapterGenerator : this.adapterGeneratorsInput.get()) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        for (Adapter adapter : this.adapters) {
            this.totalNumMutable += adapter.getNumMutable();
            this.totalNumImmutable += adapter.getNumImmutable();
        }

        this.covarianceSampler = new CenteredMultivariateNormalSampler(this.totalNumMutable);
        this.mlp = new MLP(this.totalNumMutable + this.totalNumImmutable, 1);

        this.removeResults();
    }

    @Override
    public double proposal() {
        throw new UnsupportedOperationException();
    }

    public double proposal(double oldLogLikelihood) {
        this.count++;

        if (this.count < this.BURN_IN) {
            return this.proposeAlternativeOperator();
        } else if (this.count == this.BURN_IN) {
            this.offset = oldLogLikelihood;
            System.out.println("Start to train approximate MALA with offset " + oldLogLikelihood);
        }

        this.refreshAdapters();

        // choose node to operate on for local adapters

        int nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        while (this.tree.getNode(nodeId).isLeaf() || this.tree.getNode(nodeId).isRoot()) {
            nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        }

        // get current state

        double[] oldState = this.getState(nodeId);

        // record state

        this.record(oldState, oldLogLikelihood, nodeId);

        if (this.count < this.JUST_TRAINING) {
            return this.proposeAlternativeOperator();
        } else if (this.count == this.JUST_TRAINING) {
            this.offset = oldLogLikelihood;
            System.out.println("Start approximate MALA");
        }

        // do forward pass

        double[] oldGradient = this.mlp.getGradient(oldState);
        double[] forwardMean = this.proposeMean(oldState, oldGradient);

        // update adapters and compute Jacobian correction

        double[] proposedState = this.sampleMutable(forwardMean);
        double forwardLogDensity = this.logDensity(proposedState, forwardMean);

        double oldLogJacobian = 0.0;
        double newLogJacobian = 0.0;
        double transitionCorrection = 0.0;

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;

            oldLogJacobian += adapter.getLogJacobianCorrection(nodeId);

            double[] proposedMutable = new double[adapter.getNumMutable()];
            System.arraycopy(proposedState, idx, proposedMutable, 0, adapter.getNumMutable());

            try {
                transitionCorrection += adapter.update(proposedMutable, nodeId);
            } catch (Exception e) {
                System.out.println(e);
                return Double.NEGATIVE_INFINITY;
            }

            newLogJacobian += adapter.getLogJacobianCorrection(nodeId);

            idx += adapter.getNumMutable();
        }

        // do backward pass

        double[] newState = this.getState(nodeId);
        double[] newGradient = this.mlp.getGradient(newState);
        double[] backwardMean = this.proposeMean(newState, newGradient);
        double backwardLogDensity = this.logDensity(oldState, backwardMean);

        return backwardLogDensity - forwardLogDensity + oldLogJacobian - newLogJacobian + transitionCorrection;
    }

    private void appendResult(double prediction, double logLikelihood) {
        String line = String.format(Locale.ROOT, "%f,%f%n", prediction, logLikelihood);
        try {
            Files.writeString(
                    Path.of("results.csv"),
                    line,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new IllegalStateException("failed to append MLP result", e);
        }
    }

    private void removeResults() {
        try {
            Files.deleteIfExists(Path.of("results.csv"));
        } catch (IOException e) {
            throw new IllegalStateException("failed to remove results.csv", e);
        }
    }

    private double[] getState(int nodeId) {
        double[] oldMutable = this.getMutable(nodeId);
        double[] oldImmutable = this.getImmutable(nodeId);

        double[] state = new double[oldMutable.length + oldImmutable.length];
        System.arraycopy(oldMutable, 0, state, 0, oldMutable.length);
        System.arraycopy(oldImmutable, 0, state, oldMutable.length, oldImmutable.length);

        return state;
    }

    private void record(double[] state, double logLikelihood, int nodeId) {
        double logJacobian = 0.0;

        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;
            logJacobian += adapter.getLogJacobianCorrection(nodeId);
        }

        this.mlp.record(state, new double[] {logLikelihood - this.offset});
        this.appendResult(this.mlp.runInference(state)[0] + this.offset, logLikelihood);

        double[] mutableState = new double[this.totalNumMutable];
        System.arraycopy(state, 0, mutableState, 0, mutableState.length);
        this.covarianceSampler.record(new double[] {}, mutableState);
    }

    private void refreshAdapters() {
        for (Adapter adapter : this.adapters) {
            adapter.refresh();
        }
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

    private double[] getImmutable(int nodeId) {
        double[] immutable = new double[this.totalNumImmutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterImmutable = adapter.getImmutable(nodeId);
            System.arraycopy(adapterImmutable, 0, immutable, idx, adapter.getNumImmutable());
            idx += adapter.getNumImmutable();
        }

        return immutable;
    }

    private double[] proposeMean(double[] oldState, double[] oldGradient) {
        double[] proposalMean = new double[oldState.length];

        System.arraycopy(oldState, 0, proposalMean, 0, oldState.length);

        double[] mutableGradient = new double[this.totalNumMutable];
        System.arraycopy(oldGradient, 0, mutableGradient, 0, mutableGradient.length);

        RealMatrix covariance = this.covarianceSampler.getCovariance();
        RealVector preconditionedGradient = covariance.operate(new ArrayRealVector(mutableGradient));

        for (int i = 0; i < this.totalNumMutable; i++) {
            proposalMean[i] += 0.5 * this.alpha * this.alpha * preconditionedGradient.getEntry(i);
        }

        return proposalMean;
    }

    private double[] sampleMutable(double[] state) {
        double[] noise = this.covarianceSampler.sampleConditionally(new double[] {}, this.alpha);

        double[] perturbedState = new double[state.length];
        for (int i = 0; i < state.length; i++) {
            if (i < noise.length) {
                perturbedState[i] = state[i] + noise[i];
            } else {
                perturbedState[i] = state[i];
            }
        }

        return perturbedState;
    }

    private double logDensity(double[] state, double[] mean) {
        double[] deviationFromMean = new double[this.totalNumMutable];

        for (int i = 0; i < this.totalNumMutable; i++) {
            deviationFromMean[i] = state[i] - mean[i];
        }

        return this.covarianceSampler.logDensity(new double[] {}, deviationFromMean, this.alpha);
    }


    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> nodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            nodes.addAll(adapter.listStateNodes());
        }

        return nodes;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.alpha;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("alpha must be finite and positive");
        }

        this.alpha = value;
    }

    private double proposeAlternativeOperator() {
        return this.alternativeOperators.get(Randomizer.nextInt(this.alternativeOperators.size())).proposal();
    }

    @Override
    public void optimize(double logAlpha) {
        if (this.count < this.JUST_TRAINING) return;

        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.alpha);
        this.alpha = Math.exp(delta);
    }

}
