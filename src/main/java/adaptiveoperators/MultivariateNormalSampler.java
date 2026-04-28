package adaptiveoperators;

import java.util.Arrays;
import java.util.Random;

public class MultivariateNormalSampler extends ConditionalAdaptiveSampler {

    double[] mean;
    double[][] covariance;
    int count = 0;

    private final Random rng = new Random();

    public MultivariateNormalSampler(int numConditions, int numValues) {
        super(numConditions, numValues);

        this.mean = new double[numConditions + numValues];
        this.covariance = new double[numConditions + numValues][numConditions + numValues];
    }

    @Override
    public void record(double[] conditions, double[] values) {
        int n = conditions.length + values.length;
        double[] x = new double[n];
        System.arraycopy(conditions, 0, x, 0, conditions.length);
        System.arraycopy(values, 0, x, conditions.length, values.length);

        this.count += 1;

        // delta against old mean, then update mean
        double[] delta = new double[n];
        for (int i = 0; i < n; i++) {
            delta[i] = x[i] - this.mean[i];
            this.mean[i] += delta[i] / this.count;
        }

        // delta2 against new mean, accumulate outer product into M2 (covariance)
        double[] delta2 = new double[n];
        for (int i = 0; i < n; i++) {
            delta2[i] = x[i] - this.mean[i];
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                this.covariance[i][j] += delta[i] * delta2[j];
            }
        }
    }

    @Override
    public double[] sampleConditionally(double[] conditions) {
        int nc = numConditions;
        int nv = numValues;

        // partition actual covariance (M2 / count) into sigma11, sigma12, sigma22
        double[][] s11 = new double[nc][nc];
        double[][] s12 = new double[nc][nv];
        double[][] s22 = new double[nv][nv];
        for (int i = 0; i < nc; i++)
            for (int j = 0; j < nc; j++) s11[i][j] = covariance[i][j] / count;
        for (int i = 0; i < nc; i++)
            for (int j = 0; j < nv; j++) s12[i][j] = covariance[i][nc + j] / count;
        for (int i = 0; i < nv; i++)
            for (int j = 0; j < nv; j++) s22[i][j] = covariance[nc + i][nc + j] / count;

        RealMatrix sigma12 = new Array2DRowRealMatrix(s12);
        DecompositionSolver solver11 = new CholeskyDecomposition(new Array2DRowRealMatrix(s11)).getSolver();

        // alpha = Sigma11^{-1} * (conditions - mu1)
        double[] diff = new double[nc];
        for (int i = 0; i < nc; i++) diff[i] = conditions[i] - mean[i];
        RealVector alpha = solver11.solve(new ArrayRealVector(diff));

        // conditional mean: mu2 + Sigma12^T * alpha
        RealVector condMean = new ArrayRealVector(Arrays.copyOfRange(mean, nc, nc + nv))
                .add(sigma12.transpose().operate(alpha));

        // conditional covariance: Sigma22 - Sigma12^T * Sigma11^{-1} * Sigma12
        RealMatrix condCov = new Array2DRowRealMatrix(s22)
                .subtract(sigma12.transpose().multiply(solver11.solve(sigma12)));

        // sample: condMean + L * z,  z ~ N(0, I)
        RealMatrix L = new CholeskyDecomposition(condCov).getL();
        double[] z = new double[nv];
        for (int i = 0; i < nv; i++) z[i] = rng.nextGaussian();

        return condMean.add(L.operate(new ArrayRealVector(z))).toArray();
    }

}
