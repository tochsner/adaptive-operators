package mala;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;

/**
 * Maintains the square root R of the Fisher preconditioner A = R R^T together with the
 * adaptive step size, following Titsias (2023), "Optimal Preconditioning and Fisher
 * Adaptive Langevin Sampling" (arXiv 2305.14442). All updates are O(d^2).
 */
public class FisherPreconditioner {

    private final int dimension;
    private final double rho;
    private final double targetAcceptance;

    private RealMatrix squareRoot;
    private double sigma2;

    public FisherPreconditioner(int dimension, double lambda, double rho, double targetAcceptance, double initialSigma2) {
        this.dimension = dimension;
        this.rho = rho;
        this.targetAcceptance = targetAcceptance;
        this.sigma2 = initialSigma2;

        // initializing R = (1/sqrt(lambda)) I makes eq. (13) reproduce the n = 1 case (12)
        this.squareRoot = new Array2DRowRealMatrix(dimension, dimension);
        for (int i = 0; i < dimension; i++) {
            this.squareRoot.setEntry(i, i, 1.0 / Math.sqrt(lambda));
        }
    }

    /**
     * Returns the normalized step size sigma^2 / ((1/d) tr(R R^T)).
     */
    public double sigma2R() {
        double traceAAt = 0.0;
        for (int i = 0; i < this.dimension; i++) {
            for (int j = 0; j < this.dimension; j++) {
                double entry = this.squareRoot.getEntry(i, j);
                traceAAt += entry * entry;
            }
        }

        return this.sigma2 / (traceAAt / this.dimension);
    }

    /**
     * Returns the drift increment (sigma2R / 2) R (R^T gradient).
     */
    public double[] driftIncrement(double[] gradient, double sigma2R) {
        RealVector preconditioned = this.applyPreconditioner(new ArrayRealVector(gradient));
        return preconditioned.mapMultiply(0.5 * sigma2R).toArray();
    }

    /**
     * Returns the noise increment sqrt(sigma2R) R eta for standard normal eta.
     */
    public double[] sampleNoise(double[] eta, double sigma2R) {
        return this.squareRoot.operate(new ArrayRealVector(eta)).mapMultiply(Math.sqrt(sigma2R)).toArray();
    }

    /**
     * Returns the log Hastings ratio log q(x|y) - log q(y|x) for the preconditioned MALA
     * proposal, computed without inverting A (Proposition 1):
     * 0.5 (x - y)^T (gx + gy) + (sigma2R / 8) (gx^T A gx - gy^T A gy).
     */
    public double hastings(double[] x, double[] y, double[] gradientX, double[] gradientY, double sigma2R) {
        double crossTerm = 0.0;
        for (int i = 0; i < this.dimension; i++) {
            crossTerm += (x[i] - y[i]) * (gradientX[i] + gradientY[i]);
        }

        double quadraticX = this.quadraticForm(gradientX);
        double quadraticY = this.quadraticForm(gradientY);

        return 0.5 * crossTerm + (sigma2R / 8.0) * (quadraticX - quadraticY);
    }

    /**
     * Applies the rank-one square root update (eq. 13) for the adaptation signal s:
     * phi = R^T s, r = 1 / (1 + sqrt(1 / (1 + phi^T phi))), R -= (r / (1 + phi^T phi)) (R phi) phi^T.
     */
    public void updateSquareRoot(double[] signal) {
        RealVector phi = this.squareRoot.preMultiply(new ArrayRealVector(signal));
        double phiNormSquared = phi.dotProduct(phi);

        if (!Double.isFinite(phiNormSquared) || phiNormSquared == 0.0) return;

        double r = 1.0 / (1.0 + Math.sqrt(1.0 / (1.0 + phiNormSquared)));
        RealVector rPhi = this.squareRoot.operate(phi);

        this.squareRoot = this.squareRoot.subtract(
                rPhi.mapMultiply(r / (1.0 + phiNormSquared)).outerProduct(phi));
    }

    /**
     * Applies the Robbins-Monro step size update sigma^2 *= 1 + rho (alpha - targetAcceptance).
     */
    public void updateStepSize(double alpha) {
        double updated = this.sigma2 * (1.0 + this.rho * (alpha - this.targetAcceptance));

        if (Double.isFinite(updated) && updated > 0.0) {
            this.sigma2 = updated;
        }
    }

    public double getSigma2() {
        return this.sigma2;
    }

    public void setSigma2(double sigma2) {
        if (!Double.isFinite(sigma2) || sigma2 <= 0.0) {
            throw new IllegalArgumentException("sigma2 must be finite and positive");
        }

        this.sigma2 = sigma2;
    }

    public RealMatrix getSquareRoot() {
        return this.squareRoot;
    }

    private RealVector applyPreconditioner(RealVector vector) {
        // A v = R (R^T v)
        return this.squareRoot.operate(this.squareRoot.preMultiply(vector));
    }

    private double quadraticForm(double[] gradient) {
        // g^T A g = ||R^T g||^2
        RealVector projected = this.squareRoot.preMultiply(new ArrayRealVector(gradient));
        return projected.dotProduct(projected);
    }

}
