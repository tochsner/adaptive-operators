package mala;

import adapters.*;
import adaptiveoperators.AdaptiveOperator;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.core.Pair;

import java.util.*;

public class CubeMAPGuidedMALAOperator extends AdaptiveOperator {

    public final Input<CubeAdapter> cubeAdapterInput = new Input<>("cube", "");
    public final Input<Double> betaInput = new Input<>("beta", "", 10.0);

    private CubeAdapter cubeAdapter;
    private double beta;
    private double alpha = 0.01;

    private Map<Pair<Integer, Integer>, RunningVariance> runningVariances;

    @Override
    public void initAndValidate() {
        this.cubeAdapter = this.cubeAdapterInput.get();
        this.beta = this.betaInput.get();
        this.runningVariances = new HashMap<>();
    }

    @Override
    public double proposal() {
        this.cubeAdapter.refresh();

        double[] oldCube = this.cubeAdapter.getMutable(0);
        double[] mapCube = this.cubeAdapter.getMutableMAP();

        double oldLogJacobian = this.cubeAdapter.getLogJacobianCorrection(0);

        double[] variance = recordAndGetVariance(oldCube);

        double[] forwardMean = this.proposalMean(oldCube, mapCube);
        double[] newCube = this.sampleCube(forwardMean, variance);
        double forwardLogDensity = this.getLogDensity(newCube, forwardMean, variance);

        this.cubeAdapter.update(newCube, 0);

        double newLogJacobian = this.cubeAdapter.getLogJacobianCorrection(0);
        double[] reverseMean = this.proposalMean(newCube, mapCube);
        double reverseLogDensity = this.getLogDensity(oldCube, reverseMean, variance);

        return oldLogJacobian
                + reverseLogDensity
                - newLogJacobian
                - forwardLogDensity;
    }

    private double[] recordAndGetVariance(double[] cube) {
        double[] variance = new double[cube.length];
        for (int i = 0; i < this.cubeAdapter.getCube().size() - 1; i++) {
            Integer nodeNrA = this.cubeAdapter.getCube().get(i);
            Integer nodeNrB = this.cubeAdapter.getCube().get(i + 1);

            RunningVariance runningVariance = this.runningVariances.computeIfAbsent(
                    new Pair<>(nodeNrA, nodeNrB), x -> new RunningVariance(
                            1.0, 1.0E-10
                    )
            );

            runningVariance.record(cube[i]);

            variance[i] = this.alpha * runningVariance.getVariance();
        }
        return variance;
    }

    private double[] proposalMean(double[] cube, double[] mapCube) {
        int n = this.cubeAdapter.getNumMutable();
        double[] proposalMean = new double[n];

        for (int i = 0; i < n; i++) {
            proposalMean[i] = cube[i] + 0.5 * this.alpha * this.beta * (mapCube[i] - cube[i]);
        }

        return proposalMean;
    }

    private double getLogDensity(double[] newCube, double[] forwardMean, double[] variance) {
        int n = this.cubeAdapter.getNumMutable();
        double logDensity = 0.0;

        for (int i = 0; i < n; i++) {
            double diff = newCube[i] - forwardMean[i];
            logDensity += -0.5 * (Math.log(2.0 * Math.PI * variance[i]) + diff * diff / variance[i]);
        }

        return logDensity;
    }

    private double[] sampleCube(double[] forwardMean, double[] variance) {
        int n = this.cubeAdapter.getNumMutable();
        double[] sample = new double[n];

        for (int i = 0; i < n; i++) {
            sample[i] = forwardMean[i] + Math.sqrt(variance[i]) * Randomizer.nextGaussian();
        }

        return sample;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return this.cubeAdapter.listStateNodes();
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
