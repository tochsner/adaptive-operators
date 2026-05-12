package adaptiveoperators;

import org.apache.commons.math4.legacy.linear.*;

import java.util.Arrays;
import java.util.Random;

public class MultivariateNormalSampler extends ConditionalSampler {

    double[] mean;  // [conditions, values]
    double[][] covarianceSum; // the covariance * count
    int count = 0;

    private final Random rng = new Random();

    public MultivariateNormalSampler(int numConditions, int numValues) {
        super(numConditions, numValues);

        this.mean = new double[numConditions + numValues];
        this.covarianceSum = new double[numConditions + numValues][numConditions + numValues];
    }

    @Override
    public void record(double[] conditions, double[] values) {
        int n = conditions.length + values.length;

        // we update the mean and covariances using the Welford update
        // (see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

        double[] x = new double[n];
        System.arraycopy(conditions, 0, x, 0, conditions.length);
        System.arraycopy(values, 0, x, conditions.length, values.length);

        this.count += 1;

        // old mean is needed for the Welford M2 update
        double[] oldMean = Arrays.copyOf(this.mean, n);
        for (int i = 0; i < n; i++) {
            this.mean[i] += (x[i] - this.mean[i]) / this.count;
        }

        // accumulate covariances as the outer product from old and new means
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                this.covarianceSum[i][j] += (x[i] - oldMean[i]) * (x[j] - this.mean[j]);
            }
        }
    }

    @Override
    public double[] sampleConditionally(double[] conditions) {
        ConditionalDistribution distribution = conditionalDistribution(conditions);

        // sample: condMean + L * z,  z ~ N(0, I)
        RealMatrix L = new CholeskyDecomposition(distribution.covariance).getL();
        double[] z = new double[this.numValues];
        for (int i = 0; i < this.numValues; i++) z[i] = rng.nextGaussian();

        return distribution.mean.add(L.operate(new ArrayRealVector(z))).toArray();
    }

    @Override
    public double logDensity(double[] conditions, double[] values) {
        ConditionalDistribution distribution = conditionalDistribution(conditions);
        CholeskyDecomposition decomposition = new CholeskyDecomposition(distribution.covariance);
        RealVector diff = new ArrayRealVector(values).subtract(distribution.mean);
        RealVector solved = decomposition.getSolver().solve(diff);
        double quadratic = diff.dotProduct(solved);
        double logDeterminant = 0.0;
        RealMatrix l = decomposition.getL();
        for (int i = 0; i < this.numValues; i++) {
            logDeterminant += 2.0 * Math.log(l.getEntry(i, i));
        }

        return -0.5 * (this.numValues * Math.log(2.0 * Math.PI) + logDeterminant + quadratic);
    }

    private ConditionalDistribution conditionalDistribution(double[] conditions) {
        int nc = this.numConditions;
        int nv = this.numValues;

        if (nc == 0) {
            return new ConditionalDistribution(
                    new ArrayRealVector(this.mean),
                    new Array2DRowRealMatrix(this.covarianceSum).scalarMultiply(1.0 / this.count)
            );
        }

        // partition actual covariance into sigma11, sigma12, sigma22

        double[][] s11 = new double[nc][nc];
        double[][] s12 = new double[nc][nv];
        double[][] s22 = new double[nv][nv];

        for (int i = 0; i < nc; i++)
            for (int j = 0; j < nc; j++) s11[i][j] = this.covarianceSum[i][j] / this.count;
        for (int i = 0; i < nc; i++)
            for (int j = 0; j < nv; j++) s12[i][j] = this.covarianceSum[i][nc + j] / this.count;
        for (int i = 0; i < nv; i++)
            for (int j = 0; j < nv; j++) s22[i][j] = this.covarianceSum[nc + i][nc + j] / this.count;

        RealMatrix sigma11 = new Array2DRowRealMatrix(s11);
        RealMatrix sigma12 = new Array2DRowRealMatrix(s12);
        RealMatrix sigma22 = new Array2DRowRealMatrix(s22);
        RealMatrix sigma12T = sigma12.transpose();
        DecompositionSolver solver11 = new SingularValueDecomposition(sigma11).getSolver();

        ArrayRealVector a = new ArrayRealVector(conditions);
        ArrayRealVector mu1 = new ArrayRealVector(Arrays.copyOfRange(mean, 0, nc));
        ArrayRealVector mu2 = new ArrayRealVector(Arrays.copyOfRange(mean, nc, nc + nv));

        // conditional mean: mu2 + Sigma12^T * Sigma11^{-1} * (a - mu1)
        RealVector condMean = mu2.add(sigma12T.operate(solver11.solve(a.subtract(mu1))));

        // conditional covariance: Sigma22 - Sigma12^T * Sigma11^{-1} * Sigma12
        RealMatrix condCov = sigma22.subtract(sigma12T.multiply(solver11.solve(sigma12)));

        return new ConditionalDistribution(condMean, condCov);
    }

    private static class ConditionalDistribution {
        private final RealVector mean;
        private final RealMatrix covariance;

        private ConditionalDistribution(RealVector mean, RealMatrix covariance) {
            this.mean = mean;
            this.covariance = covariance;
        }
    }

}
