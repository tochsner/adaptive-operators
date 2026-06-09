package adaptiveoperators;

import org.apache.commons.math4.legacy.linear.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CenteredMultivariateNormalSampler extends ConditionalSampler {

    double[] mean;
    double[][] covarianceSum; // the covariance * count
    int count = 0;

    int batchSize = 1;
    List<double[]> valuesBatch;

    RealMatrix covariance;
    CholeskyDecomposition choleskyDecomposition;

    private final Random rng = new Random();

    public CenteredMultivariateNormalSampler(int numValues) {
        super(0, numValues);

        this.mean = new double[numValues];
        this.covarianceSum = new double[numValues][numValues];
        this.covariance = new BlockRealMatrix(numValues, numValues);

        this.valuesBatch = new ArrayList<>();
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

        this.choleskyDecomposition = new CholeskyDecomposition(
                this.covariance.scalarMultiply(1.0), 1.0E-10, -1.0E-10
        );
    }

    @Override
    public double[] sampleConditionally(double[] conditions, double scaleFactor) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) {
            throw new RuntimeException("Non-finite conditions found.");
        }

        // sample: L * z,  z ~ N(0, I)
        RealMatrix L = this.choleskyDecomposition.getL();
        double[] z = new double[this.numValues];
        for (int i = 0; i < this.numValues; i++) z[i] = rng.nextGaussian();

        return L.operate(new ArrayRealVector(z)).toArray();
    }

    @Override
    public double logDensity(double[] conditions, double[] values, double scaleFactor) {
        RealVector diff = new ArrayRealVector(values);
        RealVector solved = this.choleskyDecomposition.getSolver().solve(diff);

        double quadratic = diff.dotProduct(solved);
        double logDeterminant = 0.0;
        RealMatrix l = this.choleskyDecomposition.getL();
        for (int i = 0; i < this.numValues; i++) {
            logDeterminant += 2.0 * Math.log(l.getEntry(i, i));
        }

        return -0.5 * (this.numValues * Math.log(2.0 * Math.PI) + logDeterminant + quadratic);
    }

}
