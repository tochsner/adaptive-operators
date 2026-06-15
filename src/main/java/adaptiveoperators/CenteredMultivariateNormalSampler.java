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

    double[] mean;
    double[][] covarianceSum; // the covariance * count
    int count = 0;

    int batchSize = 64;
    List<double[]> valuesBatch;

    RealMatrix covariance;
    RealMatrix regularizedCovariance;
    CholeskyDecomposition choleskyDecomposition;

    private final Random rng = new Random();

    public CenteredMultivariateNormalSampler(int numValues) {
        super(0, numValues);

        this.mean = new double[numValues];
        this.covarianceSum = new double[numValues][numValues];
        this.covariance = new BlockRealMatrix(numValues, numValues);

        this.valuesBatch = new ArrayList<>();
        this.choleskyDecomposition = this.regularizedCholesky(this.covariance);
    }

    @Override
    public void record(double[] conditions, double[] values) {
        if (conditions.length != 0) throw new UnsupportedOperationException("No conditions are supported.");
        if (!Arrays.stream(values).allMatch(Double::isFinite)) return;

        this.valuesBatch.add(values);

        if (this.valuesBatch.size() == this.batchSize) {
            this.processBatch();
        }
    }

    private void processBatch() {
        // update the mean and covariance using the last batch

        for (int i = 0; i < this.batchSize; i++) {
            double[] values = this.valuesBatch.get(i);

            int n = values.length;
            double[] x = values;

            // we update the mean and covariances using the Welford update
            // (see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

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
                    this.covariance.setEntry(j, k, this.covarianceSum[j][k] / this.count);
                }
            }
        }

        // reset the batch

        this.valuesBatch.clear();

        // update the solver

        this.choleskyDecomposition = this.regularizedCholesky(this.covariance);
    }

    @Override
    public double[] sampleConditionally(double[] conditions, double scaleFactor) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite conditions found.");
        }

        // sample: scaleFactor * L * z,  z ~ N(0, I), giving covariance scaleFactor^2 * Sigma
        RealMatrix L = this.choleskyDecomposition.getL();
        double[] z = new double[this.numValues];
        for (int i = 0; i < this.numValues; i++) z[i] = rng.nextGaussian();

        return L.operate(new ArrayRealVector(z)).mapMultiply(scaleFactor).toArray();
    }

    @Override
    public double logDensity(double[] conditions, double[] values, double scaleFactor) {
        // density of N(0, scaleFactor^2 * Sigma): the quadratic form is scaled by 1/scaleFactor^2
        // and the log-determinant of the covariance gains 2 * numValues * log(scaleFactor)

        RealVector diff = new ArrayRealVector(values);
        RealVector solved = this.choleskyDecomposition.getSolver().solve(diff);

        double quadratic = diff.dotProduct(solved) / (scaleFactor * scaleFactor);
        double logDeterminant = 2.0 * this.numValues * Math.log(scaleFactor);
        RealMatrix l = this.choleskyDecomposition.getL();
        for (int i = 0; i < this.numValues; i++) {
            logDeterminant += 2.0 * Math.log(l.getEntry(i, i));
        }

        return -0.5 * (this.numValues * Math.log(2.0 * Math.PI) + logDeterminant + quadratic);
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
