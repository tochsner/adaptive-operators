package adaptiveoperators;

import org.apache.commons.math4.legacy.exception.MathIllegalArgumentException;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.CholeskyDecomposition;
import org.apache.commons.math4.legacy.linear.DecompositionSolver;
import org.apache.commons.math4.legacy.linear.QRDecomposition;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;

import java.util.Arrays;
import java.util.Random;

public class GaussianMixtureSampler extends ConditionalSampler {

    static final int NUM_MIXTURES = 4;
    private static final double REGULARIZATION = 1e-8;
    private static final int INITIALIZATION_SKIP = 1_000;

    double[][] mean;
    double[][][] covarianceSum;
    double[] effectiveCounts;
    double totalEffectiveCount = 0.0;
    int count = 0;

    private final double[] initializationMean;
    private final double[][] initializationCovarianceSum;
    private int initializationCount = 0;
    private int initializedComponents = 0;
    private final Random rng = new Random();

    public GaussianMixtureSampler(int numConditions, int numValues) {
        super(numConditions, numValues);

        int dimension = numConditions + numValues;
        this.mean = new double[NUM_MIXTURES][dimension];
        this.covarianceSum = new double[NUM_MIXTURES][dimension][dimension];
        this.effectiveCounts = new double[NUM_MIXTURES];
        this.initializationMean = new double[dimension];
        this.initializationCovarianceSum = new double[dimension][dimension];
    }

    @Override
    public void record(double[] conditions, double[] values) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) return;
        if (!Arrays.stream(values).allMatch(Double::isFinite)) return;

        double[] x = join(conditions, values);
        this.count += 1;

        if (this.initializedComponents < NUM_MIXTURES) {
            updateInitializationCovariance(x);

            if (this.count == nextInitializationCount()) {
                int component = this.initializedComponents;
                System.arraycopy(x, 0, this.mean[component], 0, x.length);
                this.effectiveCounts[component] = 1.0;
                this.totalEffectiveCount += 1.0;
                this.initializedComponents += 1;

                if (this.initializedComponents == NUM_MIXTURES) {
                    initializeComponentCovariances();
                }
            }
            return;
        }

        double[] logResponsibilities = new double[NUM_MIXTURES];
        for (int k = 0; k < NUM_MIXTURES; k++) {
            logResponsibilities[k] = logWeight(k) + logGaussian(
                    new ArrayRealVector(x),
                    new ArrayRealVector(this.mean[k]),
                    covariance(k)
            );
        }

        double logNormalizer = logSumExp(logResponsibilities);
        for (int k = 0; k < NUM_MIXTURES; k++) {
            updateComponent(k, x, Math.exp(logResponsibilities[k] - logNormalizer));
        }
        this.totalEffectiveCount += 1.0;
    }

    @Override
    public double[] sampleConditionally(double[] conditions) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite conditions found.");
        }

        ensureInitialized();

        double[] logProbabilities = conditionalComponentLogProbabilities(conditions);
        double logNormalizer = logSumExp(logProbabilities);
        double u = this.rng.nextDouble();
        double cumulative = 0.0;
        int component = NUM_MIXTURES - 1;

        for (int k = 0; k < NUM_MIXTURES; k++) {
            cumulative += Math.exp(logProbabilities[k] - logNormalizer);
            if (u <= cumulative) {
                component = k;
                break;
            }
        }

        ConditionalDistribution distribution = conditionalDistribution(component, conditions);

        RealMatrix l = cholesky(distribution.covariance).getL();
        double[] z = new double[this.numValues];
        for (int i = 0; i < this.numValues; i++) z[i] = this.rng.nextGaussian();

        return distribution.mean.add(l.operate(new ArrayRealVector(z))).toArray();
    }

    @Override
    public double logDensity(double[] conditions, double[] values) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite conditions found.");
        }
        if (!Arrays.stream(values).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite values found.");
        }

        ensureInitialized();

        double[] x = join(conditions, values);
        double[] jointTerms = new double[NUM_MIXTURES];
        for (int k = 0; k < NUM_MIXTURES; k++) {
            jointTerms[k] = logWeight(k) + logGaussian(
                    new ArrayRealVector(x),
                    new ArrayRealVector(this.mean[k]),
                    covariance(k)
            );
        }

        if (this.numConditions == 0) {
            return logSumExp(jointTerms);
        }

        return logSumExp(jointTerms) - logSumExp(conditionalComponentLogProbabilities(conditions));
    }

    private int nextInitializationCount() {
        return 1 + this.initializedComponents * (INITIALIZATION_SKIP + 1);
    }

    private void updateInitializationCovariance(double[] x) {
        this.initializationCount += 1;

        double[] oldMean = Arrays.copyOf(this.initializationMean, x.length);
        for (int i = 0; i < x.length; i++) {
            this.initializationMean[i] += (x[i] - this.initializationMean[i]) / this.initializationCount;
        }

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                this.initializationCovarianceSum[i][j] +=
                        (x[i] - oldMean[i]) * (x[j] - this.initializationMean[j]);
            }
        }
    }

    private void initializeComponentCovariances() {
        for (int k = 0; k < NUM_MIXTURES; k++) {
            for (int i = 0; i < this.initializationCovarianceSum.length; i++) {
                for (int j = 0; j < this.initializationCovarianceSum[i].length; j++) {
                    this.covarianceSum[k][i][j] =
                            this.initializationCovarianceSum[i][j] / this.initializationCount;
                }
            }
        }
    }

    private double[] join(double[] conditions, double[] values) {
        double[] x = new double[conditions.length + values.length];
        System.arraycopy(conditions, 0, x, 0, conditions.length);
        System.arraycopy(values, 0, x, conditions.length, values.length);
        return x;
    }

    private void updateComponent(int component, double[] x, double responsibility) {
        if (responsibility == 0.0) return;

        double oldCount = this.effectiveCounts[component];
        double newCount = oldCount + responsibility;
        double[] oldMean = Arrays.copyOf(this.mean[component], x.length);

        for (int i = 0; i < x.length; i++) {
            this.mean[component][i] += responsibility * (x[i] - this.mean[component][i]) / newCount;
        }

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                this.covarianceSum[component][i][j] +=
                        responsibility * (x[i] - oldMean[i]) * (x[j] - this.mean[component][j]);
            }
        }

        this.effectiveCounts[component] = newCount;
    }

    private double logWeight(int component) {
        return Math.log(this.effectiveCounts[component] / this.totalEffectiveCount);
    }

    private double[] conditionalComponentLogProbabilities(double[] conditions) {
        double[] logProbabilities = new double[NUM_MIXTURES];
        for (int k = 0; k < NUM_MIXTURES; k++) {
            logProbabilities[k] = logWeight(k);
            if (this.numConditions > 0) {
                logProbabilities[k] += logGaussian(
                        new ArrayRealVector(conditions),
                        new ArrayRealVector(Arrays.copyOfRange(this.mean[k], 0, this.numConditions)),
                        conditionCovariance(k)
                );
            }
        }
        return logProbabilities;
    }

    private ConditionalDistribution conditionalDistribution(int component, double[] conditions) {
        int nc = this.numConditions;
        int nv = this.numValues;

        if (nc == 0) {
            return new ConditionalDistribution(
                    new ArrayRealVector(this.mean[component]),
                    covariance(component)
            );
        }

        RealMatrix covariance = covariance(component);
        RealMatrix sigma11 = covariance.getSubMatrix(0, nc - 1, 0, nc - 1);
        RealMatrix sigma12 = covariance.getSubMatrix(0, nc - 1, nc, nc + nv - 1);
        RealMatrix sigma22 = covariance.getSubMatrix(nc, nc + nv - 1, nc, nc + nv - 1);
        RealMatrix sigma12T = sigma12.transpose();

        DecompositionSolver solver11 = new QRDecomposition(sigma11).getSolver();
        ArrayRealVector a = new ArrayRealVector(conditions);
        ArrayRealVector mu1 = new ArrayRealVector(Arrays.copyOfRange(this.mean[component], 0, nc));
        ArrayRealVector mu2 = new ArrayRealVector(Arrays.copyOfRange(this.mean[component], nc, nc + nv));

        RealVector condMean = mu2.add(sigma12T.operate(solver11.solve(a.subtract(mu1))));
        RealMatrix condCovariance = sigma22.subtract(sigma12T.multiply(solver11.solve(sigma12)));

        return new ConditionalDistribution(condMean, condCovariance);
    }

    private RealMatrix conditionCovariance(int component) {
        RealMatrix covariance = covariance(component);
        return covariance.getSubMatrix(0, this.numConditions - 1, 0, this.numConditions - 1);
    }

    private RealMatrix covariance(int component) {
        int dimension = this.numConditions + this.numValues;
        double scale = 1.0 / this.effectiveCounts[component];
        double[][] matrix = new double[dimension][dimension];

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                matrix[i][j] = this.covarianceSum[component][i][j] * scale;
            }
            matrix[i][i] += REGULARIZATION;
        }

        return new Array2DRowRealMatrix(matrix);
    }

    private double logGaussian(RealVector x, RealVector mean, RealMatrix covariance) {
        CholeskyDecomposition decomposition = cholesky(covariance);
        RealVector diff = x.subtract(mean);
        RealVector solved = decomposition.getSolver().solve(diff);
        double quadratic = diff.dotProduct(solved);
        double logDeterminant = 0.0;
        RealMatrix l = decomposition.getL();

        for (int i = 0; i < x.getDimension(); i++) {
            logDeterminant += 2.0 * Math.log(l.getEntry(i, i));
        }

        return -0.5 * (x.getDimension() * Math.log(2.0 * Math.PI) + logDeterminant + quadratic);
    }

    private CholeskyDecomposition cholesky(RealMatrix matrix) {
        MathIllegalArgumentException lastException = null;
        double jitter = REGULARIZATION;

        for (int attempt = 0; attempt < 6; attempt++) {
            try {
                return new CholeskyDecomposition(withDiagonalJitter(matrix, attempt == 0 ? 0.0 : jitter));
            } catch (MathIllegalArgumentException e) {
                lastException = e;
                jitter *= 10.0;
            }
        }

        throw lastException;
    }

    private RealMatrix withDiagonalJitter(RealMatrix matrix, double jitter) {
        RealMatrix adjusted = matrix.add(matrix.transpose()).scalarMultiply(0.5);
        for (int i = 0; i < adjusted.getRowDimension(); i++) {
            adjusted.addToEntry(i, i, jitter);
        }
        return adjusted;
    }

    private double logSumExp(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (value > max) max = value;
        }

        double sum = 0.0;
        for (double value : values) {
            sum += Math.exp(value - max);
        }
        return max + Math.log(sum);
    }

    private void ensureInitialized() {
        if (this.initializedComponents < NUM_MIXTURES) {
            throw new RuntimeException("Gaussian mixture sampler has not initialized all components.");
        }
    }

    private static class ConditionalDistribution {
        private final RealVector mean;
        private final RealMatrix covariance;

        private ConditionalDistribution(RealVector mean, RealMatrix covariance) {
            this.mean = mean;
            this.covariance = covariance.add(covariance.transpose()).scalarMultiply(0.5);
            for (int i = 0; i < this.covariance.getRowDimension(); i++) {
                this.covariance.addToEntry(i, i, REGULARIZATION);
            }
        }
    }
}
