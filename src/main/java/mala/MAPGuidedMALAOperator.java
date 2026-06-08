package mala;

import adapters.Adapter;
import adapters.AdapterGenerator;
import adapters.MAPAdapter;
import adapters.TaxaDistanceAdapter;
import adaptiveoperators.AdaptiveOperator;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

public class MAPGuidedMALAOperator extends AdaptiveOperator {

    public final Input<List<AdapterGenerator<MAPAdapter>>> adapterGeneratorsInput = new Input<>("adapterGenerator", "", new ArrayList<>());
    public final Input<Double> betaInput = new Input<>("beta", "", 10.0);
    public final Input<Double> initialVarianceInput = new Input<>("initialVariance", "", 1.0);
    public final Input<Double> minVarianceInput = new Input<>("minVariance", "", 1.0E-12);

    private List<MAPAdapter> adapters;
    private List<AdapterGenerator<MAPAdapter>> adapterGenerators;
    private RunningVariance[] variances;
    private double beta;
    private double initialVariance;
    private double minVariance;
    private double alpha = 0.01;

    @Override
    public void initAndValidate() {
        this.adapters = new ArrayList<>();
        this.adapterGenerators = this.adapterGeneratorsInput.get();
        this.beta = this.betaInput.get();
        this.initialVariance = this.initialVarianceInput.get();
        this.minVariance = this.minVarianceInput.get();

        if (!Double.isFinite(this.beta)) {
            throw new IllegalArgumentException("beta must be finite");
        }

        if (!Double.isFinite(this.initialVariance) || this.initialVariance <= 0.0) {
            throw new IllegalArgumentException("initialVariance must be finite and positive");
        }

        if (!Double.isFinite(this.minVariance) || this.minVariance <= 0.0) {
            throw new IllegalArgumentException("minVariance must be finite and positive");
        }

        for (AdapterGenerator<MAPAdapter> adapterGenerator : this.adapterGenerators) {
            this.adapters.addAll(adapterGenerator.getAdapters());
        }

        if (this.adapters.isEmpty()) {
            throw new IllegalArgumentException("GeneralMALAOperator requires at least one MAP adapter");
        }

        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() != 1) {
                throw new IllegalArgumentException("GeneralMALAOperator requires one mutable value per MAP adapter");
            }
        }

        this.variances = new RunningVariance[this.adapters.size()];
        for (int i = 0; i < this.variances.length; i++) {
            this.variances[i] = new RunningVariance(this.initialVariance, this.minVariance);
        }
    }

    @Override
    public double proposal() {
        int adapterIdx = Randomizer.nextInt(this.adapters.size());
        MAPAdapter adapter = this.adapters.get(adapterIdx);
        adapter.refresh();

        double oldValue = adapter.getMutable(0)[0];
        double mapValue = adapter.getMutableMAP()[0];
        double oldLogJacobian = adapter.getLogJacobianCorrection(0);

        RunningVariance runningVariance = this.variances[adapterIdx];
        runningVariance.record(oldValue);

        double variance = runningVariance.getVariance();
        double proposalVariance = 2.0 * this.alpha * variance;

        if (!Double.isFinite(proposalVariance) || proposalVariance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        double forwardMean = this.proposalMean(oldValue, mapValue);
        double newValue = forwardMean + Math.sqrt(proposalVariance) * Randomizer.nextGaussian();
        double forwardLogDensity = logGaussianDensity(newValue, forwardMean, proposalVariance);

        if (!Double.isFinite(newValue) || !Double.isFinite(forwardLogDensity)) {
            return Double.NEGATIVE_INFINITY;
        }

        double transitionCorrection = adapter.update(new double[]{newValue}, 0);
        double newLogJacobian = adapter.getLogJacobianCorrection(0);
        double reverseMean = this.proposalMean(newValue, mapValue);
        double reverseLogDensity = logGaussianDensity(oldValue, reverseMean, proposalVariance);

        return oldLogJacobian
                + transitionCorrection
                + reverseLogDensity
                - newLogJacobian
                - forwardLogDensity;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }

        return stateNodes;
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

    @Override
    public void optimize(double logAlpha) {
        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.alpha);
        this.alpha = Math.exp(delta);
    }

    private double proposalMean(double value, double mapValue) {
        return value + this.alpha * this.beta * (mapValue - value);
    }

    private static double logGaussianDensity(double value, double mean, double variance) {
        double diff = value - mean;
        return -0.5 * (Math.log(2.0 * Math.PI * variance) + diff * diff / variance);
    }

    private static class RunningVariance {
        private final double initialVariance;
        private final double minVariance;
        private long count = 0;
        private double mean = 0.0;
        private double m2 = 0.0;

        private RunningVariance(double initialVariance, double minVariance) {
            this.initialVariance = initialVariance;
            this.minVariance = minVariance;
        }

        private void record(double value) {
            this.count++;
            double delta = value - this.mean;
            this.mean += delta / this.count;
            double delta2 = value - this.mean;
            this.m2 += delta * delta2;
        }

        private double getVariance() {
            if (this.count < 2) {
                return this.initialVariance;
            }

            return Math.max(this.m2 / (this.count - 1), this.minVariance);
        }
    }

}
