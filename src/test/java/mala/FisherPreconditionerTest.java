package mala;

import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.LUDecomposition;
import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class FisherPreconditionerTest {

    private static final double LAMBDA = 10.0;
    private static final double RHO = 0.015;
    private static final double TARGET_ACCEPTANCE = 0.574;

    @Test
    public void testSquareRootUpdateMatchesInverseFisher() {
        for (int dimension : new int[] {1, 3, 5}) {
            Random random = new Random(42 + dimension);
            FisherPreconditioner preconditioner = new FisherPreconditioner(
                    dimension, LAMBDA, RHO, TARGET_ACCEPTANCE, 1.0);

            // accumulate lambda I + sum of signal outer products as the reference Fisher estimate
            RealMatrix fisher = MatrixUtils.createRealIdentityMatrix(dimension).scalarMultiply(LAMBDA);

            for (int n = 0; n < 10; n++) {
                double[] signal = new double[dimension];
                for (int i = 0; i < dimension; i++) {
                    signal[i] = random.nextGaussian();
                }

                preconditioner.updateSquareRoot(signal);

                RealVector signalVector = new ArrayRealVector(signal);
                fisher = fisher.add(signalVector.outerProduct(signalVector));

                RealMatrix squareRoot = preconditioner.getSquareRoot();
                RealMatrix reconstructed = squareRoot.multiply(squareRoot.transpose());
                RealMatrix expected = new LUDecomposition(fisher).getSolver().getInverse();

                for (int i = 0; i < dimension; i++) {
                    for (int j = 0; j < dimension; j++) {
                        assertThat(reconstructed.getEntry(i, j))
                                .isCloseTo(expected.getEntry(i, j), within(1e-9));
                    }
                }
            }
        }
    }

    @Test
    public void testHastingsMatchesExplicitGaussianDensities() {
        int dimension = 4;
        Random random = new Random(7);
        FisherPreconditioner preconditioner = new FisherPreconditioner(
                dimension, LAMBDA, RHO, TARGET_ACCEPTANCE, 1.0);

        // make R non-trivial through a few rank-one updates
        for (int n = 0; n < 6; n++) {
            double[] signal = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                signal[i] = random.nextGaussian();
            }
            preconditioner.updateSquareRoot(signal);
        }

        double[] x = new double[dimension];
        double[] y = new double[dimension];
        double[] gradientX = new double[dimension];
        double[] gradientY = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            x[i] = random.nextGaussian();
            y[i] = random.nextGaussian();
            gradientX[i] = random.nextGaussian();
            gradientY[i] = random.nextGaussian();
        }

        double sigma2R = preconditioner.sigma2R();
        double actual = preconditioner.hastings(x, y, gradientX, gradientY, sigma2R);

        RealMatrix squareRoot = preconditioner.getSquareRoot();
        RealMatrix covariance = squareRoot.multiply(squareRoot.transpose()).scalarMultiply(sigma2R);

        RealVector backwardMean = new ArrayRealVector(y)
                .add(covariance.operate(new ArrayRealVector(gradientY)).mapMultiply(0.5));
        RealVector forwardMean = new ArrayRealVector(x)
                .add(covariance.operate(new ArrayRealVector(gradientX)).mapMultiply(0.5));

        double expected = logDensity(new ArrayRealVector(x), backwardMean, covariance)
                - logDensity(new ArrayRealVector(y), forwardMean, covariance);

        assertThat(actual).isCloseTo(expected, within(1e-9));
    }

    @Test
    public void testHastingsIsZeroForZeroGradients() {
        int dimension = 3;
        FisherPreconditioner preconditioner = new FisherPreconditioner(
                dimension, LAMBDA, RHO, TARGET_ACCEPTANCE, 1.0);

        double[] x = {1.0, -2.0, 0.5};
        double[] y = {-0.3, 4.0, 2.0};
        double[] zero = new double[dimension];

        assertThat(preconditioner.hastings(x, y, zero, zero, preconditioner.sigma2R())).isEqualTo(0.0);
    }

    @Test
    public void testStepSizeAdaptsTowardsTargetAcceptance() {
        FisherPreconditioner preconditioner = new FisherPreconditioner(
                2, LAMBDA, RHO, TARGET_ACCEPTANCE, 1.0);

        double initial = preconditioner.getSigma2();

        preconditioner.updateStepSize(0.9);
        assertThat(preconditioner.getSigma2()).isGreaterThan(initial);

        double increased = preconditioner.getSigma2();

        preconditioner.updateStepSize(0.1);
        assertThat(preconditioner.getSigma2()).isLessThan(increased);
        assertThat(preconditioner.getSigma2()).isPositive();
    }

    private static double logDensity(RealVector value, RealVector mean, RealMatrix covariance) {
        int dimension = value.getDimension();
        LUDecomposition decomposition = new LUDecomposition(covariance);

        RealVector centered = value.subtract(mean);
        RealVector solved = decomposition.getSolver().solve(centered);

        return -0.5 * dimension * Math.log(2.0 * Math.PI)
                - 0.5 * Math.log(decomposition.getDeterminant())
                - 0.5 * centered.dotProduct(solved);
    }

}
