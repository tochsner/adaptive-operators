package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.spec.inference.parameter.RealScalarParam;

import java.util.LinkedList;
import java.util.function.Function;

public class LocalTreeTransport {

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
    }

    private double gaussianCDF(double o) {
        // compute G for the Standard univariate Gaussian
    }

    private double gaussianLogPDF(double[] currentTransportedState) {
        // compute log g for the Standard multivariate Gaussian
    }

    private double felsensteinCDF(int i, double[] distances) {
        // compute F(d_i | d_0 ... d_{i-1}) for the local Felsenstein likelihood
    }

    private double felsensteinLogPDF(double[] distances) {
        // compute the joint log f(d_1 ... d_i) for the local Felsenstein likelihood
    }

    private double findRoot(Function<Double, Double> function) {

    }

}
