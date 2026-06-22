package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.spec.inference.parameter.RealScalarParam;
import org.apache.commons.math4.legacy.analysis.UnivariateFunction;
import org.apache.commons.math4.legacy.analysis.solvers.BrentSolver;
import org.apache.commons.statistics.distribution.NormalDistribution;

import java.util.LinkedList;
import java.util.function.Function;

public class LocalTreeTransport {

    private static final int MAX_ROOT_EVALUATIONS = 1000;
    private static final int MAX_BRACKET_EXPANSIONS = 60;
    private static final double ROOT_RELATIVE_ACCURACY = 1.0E-10;
    private static final double ROOT_ABSOLUTE_ACCURACY = 1.0E-12;
    private static final double ROOT_FUNCTION_ACCURACY = 1.0E-12;
    private static final double INITIAL_ROOT_UPPER_BOUND = 1.0;
    private static final double MIN_CDF_PROBABILITY = 1.0E-15;
    private static final double MAX_CDF_PROBABILITY = 1.0 - MIN_CDF_PROBABILITY;
    private static final NormalDistribution STANDARD_NORMAL = NormalDistribution.of(0.0, 1.0);

    int k;
    LinkedList<Integer> cube;
    Alignment alignment;
    RealScalarParam<?> clockRate;

    public LocalTreeTransport(int k, LinkedList<Integer> cube, Alignment alignment, RealScalarParam<?> clockRate) {
        this.k = k;
        this.cube = cube;
        this.alignment = alignment;
        this.clockRate = clockRate;
    }

    public double[] transport(double[] distances) {
        // transport the k distances to the multivariate Gaussian space

        int windowSize = distances.length;

        double[] transported = new double[windowSize];
        for (int i = 0; i < windowSize; i++) {
            transported[i] = this.invGaussianCDF(
                    this.felsensteinCDF(i, distances)
            );
        }

        return transported;
    }

    public double[] transportBack(double[] transportedState) {
        // transport transportedState in the multivariate Gaussian space back to k distances

        int windowSize = transportedState.length;

        double[] distances = new double[windowSize];
        for (int i = 0; i < windowSize; i++) {
            final int tempI = i;
            distances[i] = this.findRoot(
                    x -> {
                        distances[tempI] = x;
                        return this.felsensteinCDF(tempI, distances) - this.gaussianCDF(transportedState[tempI]);
                    }
            );
        }

        return distances;
    }

    public double getTransportCorrection(
            double[] currentDistances, double[] newDistances, double[] currentTransportedState, double[] newTransportedState
    ) {
        // compute \log l_F(d) - \log l_F(d^\ast) - \log \varphi(z) + \log \varphi(z^\ast)
        return this.felsensteinLogPDF(currentDistances)
                - this.felsensteinLogPDF(newDistances)
                - this.gaussianLogPDF(currentTransportedState)
                + this.gaussianLogPDF(newTransportedState);
    }

    private double invGaussianCDF(double o) {
        // compute G^{-1} for the Standard univariate Gaussian
        double probability = Math.min(MAX_CDF_PROBABILITY, Math.max(MIN_CDF_PROBABILITY, o));
        return STANDARD_NORMAL.inverseCumulativeProbability(probability);
    }

    private double gaussianCDF(double o) {
        // compute G for the Standard univariate Gaussian
        return STANDARD_NORMAL.cumulativeProbability(o);
    }

    private double gaussianLogPDF(double[] currentTransportedState) {
        // compute log g for the Standard multivariate Gaussian
        double logPDF = 0.0;
        for (double value : currentTransportedState) {
            logPDF += STANDARD_NORMAL.logDensity(value);
        }
        return logPDF;
    }

    private double felsensteinCDF(int i, double[] distances) {
        // compute F(d_i | d_0 ... d_{i-1}) for the local Felsenstein likelihood
    }

    private double felsensteinLogPDF(double[] distances) {
        // compute the joint log f(d_1 ... d_i) for the local Felsenstein likelihood
    }

    private double findRoot(Function<Double, Double> function) {
        double lower = 0.0;
        double lowerValue = function.apply(lower);
        if (!Double.isFinite(lowerValue)) {
            throw new IllegalArgumentException("root function must be finite at the lower bound");
        }
        if (lowerValue == 0.0) {
            return lower;
        }

        double upper = INITIAL_ROOT_UPPER_BOUND;
        double upperValue = function.apply(upper);
        if (!Double.isFinite(upperValue)) {
            throw new IllegalArgumentException("root function must be finite at the initial upper bound");
        }

        int expansions = 0;
        while (Math.signum(lowerValue) == Math.signum(upperValue) && expansions < MAX_BRACKET_EXPANSIONS) {
            upper *= 2.0;
            upperValue = function.apply(upper);
            if (!Double.isFinite(upperValue)) {
                throw new IllegalArgumentException("root function became non-finite while expanding the bracket");
            }
            expansions++;
        }

        if (Math.signum(lowerValue) == Math.signum(upperValue)) {
            throw new IllegalArgumentException("could not bracket root for distance transform");
        }

        UnivariateFunction univariateFunction = function::apply;
        BrentSolver solver = new BrentSolver(
                ROOT_RELATIVE_ACCURACY,
                ROOT_ABSOLUTE_ACCURACY,
                ROOT_FUNCTION_ACCURACY
        );
        return solver.solve(MAX_ROOT_EVALUATIONS, univariateFunction, lower, upper);
    }

}
