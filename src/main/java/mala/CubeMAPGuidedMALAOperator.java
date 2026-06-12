package mala;

import adapters.*;
import adaptiveoperators.AdaptiveOperator;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.core.Pair;

import java.util.*;

/**
 * MAP-guided MALA operator for a {@link adapters.CubeAdapter}'s vector of coordinates. Like
 * {@link MAPGuidedMALAOperator} the deterministic forward step pulls each coordinate towards
 * its MAP value, but here all coordinates are proposed together with independent Gaussian
 * noise whose per-coordinate variance is a running estimate kept per cube edge (a diagonal
 * Gaussian proposal). The proposal plugs into the shared {@link GaussianProposal} mechanics
 * through an inner kernel.
 */
public class CubeMAPGuidedMALAOperator extends AdaptiveOperator {

    public final Input<CubeAdapter> cubeAdapterInput = new Input<>("cube", "");
    public final Input<Double> betaInput = new Input<>("beta", "", 1.0);

    private CubeAdapter cubeAdapter;
    private double beta;
    private double alpha = 1.0;

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

        double[] variance = recordAndGetVariance(oldCube);

        GaussianProposalKernel kernel = new Kernel(mapCube, variance);

        return GaussianProposal.propose(
                oldCube,
                kernel,
                proposed -> {
                    double oldLogJacobian = this.cubeAdapter.getLogJacobianCorrection(0);
                    this.cubeAdapter.update(proposed, 0);
                    double newLogJacobian = this.cubeAdapter.getLogJacobianCorrection(0);
                    return oldLogJacobian - newLogJacobian;
                }).logHastingsRatio();
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

    /**
     * MAP-guided diagonal kernel: the drift pulls each cube coordinate towards its MAP value
     * and the noise is an independent Gaussian per coordinate with the running variance.
     */
    private final class Kernel extends AbstractDensityKernel {

        private final double[] mapCube;
        private final double[] variance;

        private Kernel(double[] mapCube, double[] variance) {
            this.mapCube = mapCube;
            this.variance = variance;
        }

        @Override
        public double[] mean(double[] point) {
            double[] mean = new double[point.length];
            for (int i = 0; i < point.length; i++) {
                mean[i] = point[i] + 0.5 * alpha * beta * (this.mapCube[i] - point[i]);
            }
            return mean;
        }

        @Override
        public double[] sampleNoise() {
            double[] noise = new double[this.variance.length];
            for (int i = 0; i < noise.length; i++) {
                noise[i] = Math.sqrt(this.variance[i]) * Randomizer.nextGaussian();
            }
            return noise;
        }

        @Override
        public double logDensity(double[] point, double[] mean) {
            double logDensity = 0.0;
            for (int i = 0; i < point.length; i++) {
                double diff = point[i] - mean[i];
                logDensity += -0.5 * (Math.log(2.0 * Math.PI * this.variance[i]) + diff * diff / this.variance[i]);
            }
            return logDensity;
        }
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
