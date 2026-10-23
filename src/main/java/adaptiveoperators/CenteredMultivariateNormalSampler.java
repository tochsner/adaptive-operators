package adaptiveoperators;

import org.apache.commons.math4.legacy.linear.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CenteredMultivariateNormalSampler extends ConditionalSampler {

    private static final double SYMMETRY_THRESHOLD = 1.0E-10;
    private static final double POSITIVITY_THRESHOLD = 1.0E-12;
    private static final double MIN_JITTER = 1.0E-12;
    private static final double RELATIVE_JITTER = 1.0E-10;
    private static final int MAX_CHOLESKY_ATTEMPTS = 12;

    double[] mean;  // [conditions, values]
    double[][] covarianceSum; // the joint covariance * count
    int count = 0;

    int batchSize = 64;
    List<double[]> conditionsBatch;
    List<double[]> valuesBatch;

    RealMatrix regressionMatrix; // Sigma11^{-1} * Sigma12, maps centered conditions to the conditional mean
    RealMatrix regularizedCovariance; // the conditional covariance of the values
    CholeskyDecomposition choleskyDecomposition;

    private final Random rng = new Random();

    public CenteredMultivariateNormalSampler(int numValues) {
        this(0, numValues);
    }

    public CenteredMultivariateNormalSampler(int numConditions, int numValues) {
        super(numConditions, numValues);

        int n = numConditions + numValues;
        this.mean = new double[n];
        this.covarianceSum = new double[n][n];

        this.conditionsBatch = new ArrayList<>();
        this.valuesBatch = new ArrayList<>();

        // without conditions, there is nothing to regress on, and a matrix without rows cannot be built
        this.regressionMatrix = numConditions > 0 ? new BlockRealMatrix(numConditions, numValues) : null;
        this.choleskyDecomposition = this.regularizedCholesky(new BlockRealMatrix(numValues, numValues));
    }

    @Override
    public void record(double[] conditions, double[] values) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) return;
        if (!Arrays.stream(values).allMatch(Double::isFinite)) return;

        this.conditionsBatch.add(conditions);
        this.valuesBatch.add(values);

        if (this.valuesBatch.size() == this.batchSize) {
            this.processBatch();
        }
    }

    private void processBatch() {
        // update the mean and covariance using the last batch

        for (int i = 0; i < this.batchSize; i++) {
            double[] conditions = this.conditionsBatch.get(i);
            double[] values = this.valuesBatch.get(i);

            int n = conditions.length + values.length;

            // we update the mean and covariances using the Welford update
            // (see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

            double[] x = new double[n];
            System.arraycopy(conditions, 0, x, 0, conditions.length);
            System.arraycopy(values, 0, x, conditions.length, values.length);

            this.count += 1;

            // old mean is needed for the Welford M2 update
            double[] oldMean = Arrays.copyOf(this.mean, n);
            for (int j = 0; j < n; j++) {
                this.mean[j] += (x[j] - this.mean[j]) / this.count;
            }

            // accumulate covariances as the outer product from old and new means
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    this.covarianceSum[j][k] += (x[j] - oldMean[j]) * (x[k] - this.mean[k]);
                }
            }
        }

        // reset the batch

        this.conditionsBatch.clear();
        this.valuesBatch.clear();

        this.updateConditionalDistribution();
    }

    private void updateConditionalDistribution() {
        int nc = this.numConditions;
        int nv = this.numValues;

        // partition the covariance into sigma11, sigma12, sigma22

        double[][] s11 = new double[nc][nc];
        double[][] s12 = new double[nc][nv];
        double[][] s22 = new double[nv][nv];

        for (int i = 0; i < nc; i++)
            for (int j = 0; j < nc; j++) s11[i][j] = this.covarianceSum[i][j] / this.count;
        for (int i = 0; i < nc; i++)
            for (int j = 0; j < nv; j++) s12[i][j] = this.covarianceSum[i][nc + j] / this.count;
        for (int i = 0; i < nv; i++)
            for (int j = 0; j < nv; j++) s22[i][j] = this.covarianceSum[nc + i][nc + j] / this.count;

        RealMatrix sigma22 = new Array2DRowRealMatrix(s22);

        if (nc == 0) {
            this.choleskyDecomposition = this.regularizedCholesky(sigma22);
            return;
        }

        RealMatrix sigma12 = new Array2DRowRealMatrix(s12);
        DecompositionSolver solver11 = new SingularValueDecomposition(new Array2DRowRealMatrix(s11)).getSolver();

        // conditional covariance: Sigma22 - Sigma12^T * Sigma11^{-1} * Sigma12
        this.regressionMatrix = solver11.solve(sigma12);
        RealMatrix conditionalCovariance = sigma22.subtract(sigma12.transpose().multiply(this.regressionMatrix));

        this.choleskyDecomposition = this.regularizedCholesky(conditionalCovariance);
    }

    @Override
    public double[] sampleConditionally(double[] conditions, double scaleFactor) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite conditions found.");
        }

        // sample: condMean + scaleFactor * L * z,  z ~ N(0, I), giving covariance scaleFactor^2 * Sigma
        RealMatrix L = this.choleskyDecomposition.getL();
        double[] z = new double[this.numValues];
        for (int i = 0; i < this.numValues; i++) z[i] = rng.nextGaussian();

        return this.computeConditionalMean(conditions)
                .add(L.operate(new ArrayRealVector(z)).mapMultiply(scaleFactor))
                .toArray();
    }

    @Override
    public double logDensity(double[] conditions, double[] values, double scaleFactor) {
        // density of N(condMean, scaleFactor^2 * Sigma): the quadratic form is scaled by 1/scaleFactor^2
        // and the log-determinant of the covariance gains 2 * numValues * log(scaleFactor)

        RealVector diff = new ArrayRealVector(values).subtract(this.computeConditionalMean(conditions));
        RealVector solved = this.choleskyDecomposition.getSolver().solve(diff);

        double quadratic = diff.dotProduct(solved) / (scaleFactor * scaleFactor);
        double logDeterminant = 2.0 * this.numValues * Math.log(scaleFactor);
        RealMatrix l = this.choleskyDecomposition.getL();
        for (int i = 0; i < this.numValues; i++) {
            logDeterminant += 2.0 * Math.log(l.getEntry(i, i));
        }

        return -0.5 * (this.numValues * Math.log(2.0 * Math.PI) + logDeterminant + quadratic);
    }

    /** Computes Sigma12^T * Sigma11^{-1} * (conditions - mu1), i.e. the conditional mean without the value mean mu2. */
    private RealVector computeConditionalMean(double[] conditions) {
        if (this.numConditions == 0) return new ArrayRealVector(this.numValues);

        RealVector centered = new ArrayRealVector(conditions)
                .subtract(new ArrayRealVector(Arrays.copyOfRange(this.mean, 0, this.numConditions)));
        return this.regressionMatrix.preMultiply(centered);
    }

    public RealMatrix getCovariance() {
        return this.regularizedCovariance;
    }

    public double[] getMean() {
        return Arrays.copyOf(this.mean, this.mean.length);
    }

    private CholeskyDecomposition regularizedCholesky(RealMatrix matrix) {
        RealMatrix symmetricMatrix = symmetrized(matrix);
        double jitter = initialJitter(symmetricMatrix);

        for (int attempt = 0; attempt < MAX_CHOLESKY_ATTEMPTS; attempt++) {
            RealMatrix regularizedMatrix = addDiagonal(symmetricMatrix, jitter);
            try {
                CholeskyDecomposition decomposition = new CholeskyDecomposition(
                        regularizedMatrix, SYMMETRY_THRESHOLD, POSITIVITY_THRESHOLD
                );
                this.regularizedCovariance = regularizedMatrix;
                return decomposition;
            } catch (RuntimeException ignored) {
                jitter *= 10.0;
            }
        }

        RealMatrix regularizedMatrix = addDiagonal(symmetricMatrix, jitter);
        CholeskyDecomposition decomposition = new CholeskyDecomposition(
                regularizedMatrix, SYMMETRY_THRESHOLD, POSITIVITY_THRESHOLD
        );
        this.regularizedCovariance = regularizedMatrix;
        return decomposition;
    }

    private static RealMatrix symmetrized(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int columns = matrix.getColumnDimension();
        RealMatrix symmetricMatrix = new BlockRealMatrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = i; j < columns; j++) {
                double value = 0.5 * (matrix.getEntry(i, j) + matrix.getEntry(j, i));
                symmetricMatrix.setEntry(i, j, value);
                symmetricMatrix.setEntry(j, i, value);
            }
        }

        return symmetricMatrix;
    }

    private static double initialJitter(RealMatrix matrix) {
        double scale = 0.0;

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                scale = Math.max(scale, Math.abs(matrix.getEntry(i, j)));
            }
        }

        return Math.max(MIN_JITTER, scale * RELATIVE_JITTER);
    }

    private static RealMatrix addDiagonal(RealMatrix matrix, double value) {
        RealMatrix regularizedMatrix = matrix.copy();

        for (int i = 0; i < regularizedMatrix.getRowDimension(); i++) {
            regularizedMatrix.addToEntry(i, i, value);
        }

        return regularizedMatrix;
    }

}
