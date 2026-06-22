package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.spec.inference.parameter.RealScalarParam;
import org.apache.commons.statistics.distribution.NormalDistribution;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LocalTreeTransport {

    private static final int GRID_SIZE = 15;
    private static final int LAST_GRID_INDEX = GRID_SIZE - 1;
    private static final double MIN_CDF_PROBABILITY = 1.0E-15;
    private static final double MAX_CDF_PROBABILITY = 1.0 - MIN_CDF_PROBABILITY;
    private static final double MIN_LOG_DENSITY = 1.0E-300;
    private static final NormalDistribution STANDARD_NORMAL = NormalDistribution.of(0.0, 1.0);
    private static final Map<GridCacheKey, GridData> GRID_CACHE = new HashMap<>();

    private final double maxDistance;
    private final double gridSpacing;
    private final double[] grid;

    private final double[][] weightGrid;
    private final double[][] cdf1GivenD0Grid;
    private final double[] cdf0;
    private final double maxLogLikelihood;
    private final double totalMass;

    public LocalTreeTransport(List<String> taxonIds, Alignment alignment, RealScalarParam<?> clockRate, double maxDistance) {
        if (!Double.isFinite(maxDistance) || maxDistance <= 0.0) {
            throw new IllegalArgumentException("maxDistance must be finite and positive");
        }

        GridData gridData = this.getOrBuildGridData(taxonIds, alignment, clockRate, maxDistance);

        this.maxDistance = gridData.maxDistance;
        this.gridSpacing = gridData.gridSpacing;
        this.grid = gridData.grid;
        this.weightGrid = gridData.weightGrid;
        this.cdf1GivenD0Grid = gridData.cdf1GivenD0Grid;
        this.cdf0 = gridData.cdf0;
        this.maxLogLikelihood = gridData.maxLogLikelihood;
        this.totalMass = gridData.totalMass;

        if (!Double.isFinite(this.totalMass) || this.totalMass <= 0.0) {
            throw new IllegalArgumentException("grid likelihood mass must be finite and positive");
        }
    }

    public double[] transport(double[] distances) {
        this.validateDistanceDimension(distances);

        double[] transported = new double[distances.length];
        transported[0] = this.invGaussianCDF(this.felsensteinCDF(0, distances));
        transported[1] = this.invGaussianCDF(this.felsensteinCDF(1, distances));

        if (Arrays.stream(transported).anyMatch(Double::isInfinite)) {
            throw new IllegalStateException("transport produced infinite Gaussian coordinates");
        }

        return transported;
    }

    public double[] transportBack(double[] transportedState) {
        this.validateDistanceDimension(transportedState);

        double[] distances = new double[transportedState.length];
        distances[0] = this.inverseCDF0(this.gaussianCDF(transportedState[0]));
        distances[1] = this.inverseCDF1(distances[0], this.gaussianCDF(transportedState[1]));

        return distances;
    }

    public double getTransportCorrection(
            double[] currentDistances, double[] newDistances, double[] currentTransportedState, double[] newTransportedState
    ) {
        return this.felsensteinLogPDF(currentDistances)
                - this.felsensteinLogPDF(newDistances)
                - this.gaussianLogPDF(currentTransportedState)
                + this.gaussianLogPDF(newTransportedState);
    }

    private double invGaussianCDF(double o) {
        double probability = Math.min(MAX_CDF_PROBABILITY, Math.max(MIN_CDF_PROBABILITY, o));
        return STANDARD_NORMAL.inverseCumulativeProbability(probability);
    }

    private double gaussianCDF(double o) {
        return STANDARD_NORMAL.cumulativeProbability(o);
    }

    private double gaussianLogPDF(double[] transportedState) {
        double logPDF = 0.0;
        for (double value : transportedState) {
            logPDF += STANDARD_NORMAL.logDensity(value);
        }
        return logPDF;
    }

    private double felsensteinCDF(int i, double[] distances) {
        this.validateDistanceDimension(distances);

        if (i == 0) {
            return this.evaluateCDF0(distances[0]);
        } else if (i == 1) {
            return this.evaluateCDF1(distances[0], distances[1]);
        }

        throw new IllegalArgumentException("only two distance dimensions are supported");
    }

    private double felsensteinLogPDF(double[] distances) {
        this.validateDistanceDimension(distances);

        double clamped0 = this.clampDistance(distances[0]);
        double clamped1 = this.clampDistance(distances[1]);
        double scaledWeight = Math.max(MIN_LOG_DENSITY, this.bilinearInterpolate(this.weightGrid, clamped0, clamped1));
        return this.maxLogLikelihood + Math.log(scaledWeight);
    }

    private static synchronized GridData getOrBuildGridData(
            List<String> taxonIds,
            Alignment alignment,
            RealScalarParam<?> clockRate,
            double maxDistance
    ) {
        GridCacheKey key = new GridCacheKey(taxonIds, System.identityHashCode(alignment), getClockRateValue(clockRate));
        GridData cached = GRID_CACHE.get(key);
        if (cached != null && cached.maxDistance >= maxDistance) {
            return cached;
        }

        GridData gridData = buildGridData(taxonIds, alignment, clockRate, maxDistance);
        GRID_CACHE.put(key, gridData);
        return gridData;
    }

    private static double getClockRateValue(RealScalarParam<?> clockRate) {
        return clockRate == null ? 1.0 : clockRate.get();
    }

    private static GridData buildGridData(
            List<String> taxonIds,
            Alignment alignment,
            RealScalarParam<?> clockRate,
            double maxDistance
    ) {
        double gridSpacing = maxDistance / LAST_GRID_INDEX;
        double[] grid = buildGrid(gridSpacing);
        double[][] logLikelihoodGrid = new double[GRID_SIZE][GRID_SIZE];
        Approximate3TaxaFelsenstein approximateFelsenstein =
                new Approximate3TaxaFelsenstein(taxonIds, alignment, clockRate);

        double maxLogLikelihood = populateLogLikelihoodGrid(approximateFelsenstein, grid, logLikelihoodGrid);
        double[][] weightGrid = buildWeightGrid(logLikelihoodGrid, maxLogLikelihood);
        double[][] cdf1GivenD0Grid = buildConditionalCDF1Grid(weightGrid, gridSpacing);
        double[] marginal0 = buildMarginal0(cdf1GivenD0Grid);
        double[] cdf0 = buildCDF0(marginal0, gridSpacing);
        double totalMass = cdf0[LAST_GRID_INDEX];

        return new GridData(
                maxDistance,
                gridSpacing,
                grid,
                logLikelihoodGrid,
                weightGrid,
                cdf1GivenD0Grid,
                marginal0,
                cdf0,
                maxLogLikelihood,
                totalMass
        );
    }

    private static double[] buildGrid(double gridSpacing) {
        double[] grid = new double[GRID_SIZE];
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i] = i * gridSpacing;
        }
        return grid;
    }

    private static double populateLogLikelihoodGrid(
            Approximate3TaxaFelsenstein approximateFelsenstein,
            double[] grid,
            double[][] logLikelihoodGrid
    ) {
        double maxLogLikelihood = Double.NEGATIVE_INFINITY;
        double[] distances = new double[2];

        for (int i = 0; i < GRID_SIZE; i++) {
            distances[0] = grid[i];
            for (int j = 0; j < GRID_SIZE; j++) {
                distances[1] = grid[j];
                double logLikelihood = approximateFelsenstein.getApproximateLogFelsenstein(distances);
                logLikelihoodGrid[i][j] = logLikelihood;
                if (Double.isFinite(logLikelihood)) {
                    maxLogLikelihood = Math.max(maxLogLikelihood, logLikelihood);
                }
            }
        }

        if (!Double.isFinite(maxLogLikelihood)) {
            throw new IllegalArgumentException("grid contains no finite log likelihood values");
        }
        return maxLogLikelihood;
    }

    private static double[][] buildWeightGrid(double[][] logLikelihoodGrid, double maxLogLikelihood) {
        double[][] weights = new double[GRID_SIZE][GRID_SIZE];

        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                double logLikelihood = logLikelihoodGrid[i][j];
                weights[i][j] = Double.isFinite(logLikelihood)
                        ? Math.exp(logLikelihood - maxLogLikelihood)
                        : 0.0;
            }
        }

        return weights;
    }

    private static double[][] buildConditionalCDF1Grid(double[][] weightGrid, double gridSpacing) {
        double[][] cdf = new double[GRID_SIZE][GRID_SIZE];

        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 1; j < GRID_SIZE; j++) {
                cdf[i][j] = cdf[i][j - 1]
                        + trapezoidArea(weightGrid[i][j - 1], weightGrid[i][j], gridSpacing);
            }
        }

        return cdf;
    }

    private static double[] buildMarginal0(double[][] cdf1GivenD0Grid) {
        double[] marginal = new double[GRID_SIZE];
        for (int i = 0; i < GRID_SIZE; i++) {
            marginal[i] = cdf1GivenD0Grid[i][LAST_GRID_INDEX];
        }
        return marginal;
    }

    private static double[] buildCDF0(double[] marginal0, double gridSpacing) {
        double[] cdf = new double[GRID_SIZE];
        for (int i = 1; i < GRID_SIZE; i++) {
            cdf[i] = cdf[i - 1] + trapezoidArea(marginal0[i - 1], marginal0[i], gridSpacing);
        }
        return cdf;
    }

    private double evaluateCDF0(double distance0) {
        double cdf = this.linearInterpolate(this.grid, this.cdf0, this.clampDistance(distance0)) / this.totalMass;
        return this.clampProbability(cdf);
    }

    private double evaluateCDF1(double distance0, double distance1) {
        double clamped0 = this.clampDistance(distance0);
        double clamped1 = this.clampDistance(distance1);
        GridPosition position0 = this.getGridPosition(clamped0);

        double numeratorLower = this.linearInterpolate(this.grid, this.cdf1GivenD0Grid[position0.lowerIndex], clamped1);
        double numeratorUpper = this.linearInterpolate(this.grid, this.cdf1GivenD0Grid[position0.upperIndex], clamped1);
        double numerator = this.interpolate(numeratorLower, numeratorUpper, position0.fraction);

        double denominator = this.interpolate(
                this.cdf1GivenD0Grid[position0.lowerIndex][LAST_GRID_INDEX],
                this.cdf1GivenD0Grid[position0.upperIndex][LAST_GRID_INDEX],
                position0.fraction
        );

        if (denominator <= 0.0) {
            return MIN_CDF_PROBABILITY;
        }

        return this.clampProbability(numerator / denominator);
    }

    private double inverseCDF0(double probability) {
        double target = this.clampProbability(probability) * this.totalMass;
        return this.inverseMonotone(this.grid, this.cdf0, target);
    }

    private double inverseCDF1(double distance0, double probability) {
        double clamped0 = this.clampDistance(distance0);
        GridPosition position0 = this.getGridPosition(clamped0);
        double[] conditionalCDF = new double[GRID_SIZE];

        for (int j = 0; j < GRID_SIZE; j++) {
            conditionalCDF[j] = this.interpolate(
                    this.cdf1GivenD0Grid[position0.lowerIndex][j],
                    this.cdf1GivenD0Grid[position0.upperIndex][j],
                    position0.fraction
            );
        }

        double totalConditionalMass = conditionalCDF[LAST_GRID_INDEX];
        if (totalConditionalMass <= 0.0) {
            return 0.0;
        }

        double target = this.clampProbability(probability) * totalConditionalMass;
        return this.inverseMonotone(this.grid, conditionalCDF, target);
    }

    private double bilinearInterpolate(double[][] values, double x, double y) {
        GridPosition xPosition = this.getGridPosition(x);
        GridPosition yPosition = this.getGridPosition(y);

        double lowerY = this.interpolate(
                values[xPosition.lowerIndex][yPosition.lowerIndex],
                values[xPosition.lowerIndex][yPosition.upperIndex],
                yPosition.fraction
        );
        double upperY = this.interpolate(
                values[xPosition.upperIndex][yPosition.lowerIndex],
                values[xPosition.upperIndex][yPosition.upperIndex],
                yPosition.fraction
        );
        return this.interpolate(lowerY, upperY, xPosition.fraction);
    }

    private double linearInterpolate(double[] xValues, double[] yValues, double x) {
        GridPosition position = this.getGridPosition(x);
        return this.interpolate(yValues[position.lowerIndex], yValues[position.upperIndex], position.fraction);
    }

    private double inverseMonotone(double[] xValues, double[] yValues, double y) {
        if (y <= yValues[0]) {
            return xValues[0];
        }
        if (y >= yValues[LAST_GRID_INDEX]) {
            return xValues[LAST_GRID_INDEX];
        }

        int lower = 0;
        int upper = LAST_GRID_INDEX;
        while (upper - lower > 1) {
            int midpoint = (lower + upper) / 2;
            if (yValues[midpoint] < y) {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }

        double span = yValues[upper] - yValues[lower];
        if (span <= 0.0) {
            return xValues[lower];
        }

        double fraction = (y - yValues[lower]) / span;
        return this.interpolate(xValues[lower], xValues[upper], fraction);
    }

    private GridPosition getGridPosition(double value) {
        double clamped = this.clampDistance(value);
        int lowerIndex = (int) Math.floor(clamped / this.gridSpacing);

        if (lowerIndex >= LAST_GRID_INDEX) {
            return new GridPosition(LAST_GRID_INDEX, LAST_GRID_INDEX, 0.0);
        }

        int upperIndex = lowerIndex + 1;
        double fraction = (clamped - this.grid[lowerIndex]) / this.gridSpacing;
        return new GridPosition(lowerIndex, upperIndex, fraction);
    }

    private double trapezoidArea(double leftHeight, double rightHeight) {
        return 0.5 * (leftHeight + rightHeight) * this.gridSpacing;
    }

    private static double trapezoidArea(double leftHeight, double rightHeight, double gridSpacing) {
        return 0.5 * (leftHeight + rightHeight) * gridSpacing;
    }

    private double interpolate(double lower, double upper, double fraction) {
        return lower + fraction * (upper - lower);
    }

    private double clampDistance(double distance) {
        if (!Double.isFinite(distance)) {
            throw new IllegalArgumentException("distance must be finite");
        }
        return Math.min(this.maxDistance, Math.max(0.0, distance));
    }

    private double clampProbability(double probability) {
        return Math.min(MAX_CDF_PROBABILITY, Math.max(MIN_CDF_PROBABILITY, probability));
    }

    private void validateDistanceDimension(double[] distances) {
        if (distances.length != 2) {
            throw new IllegalArgumentException("grid transport requires exactly two distance dimensions");
        }
    }

    private record GridPosition(int lowerIndex, int upperIndex, double fraction) {
    }

    private record GridCacheKey(List<String> taxonIds, int alignmentIdentity, double clockRate) {

        private GridCacheKey(List<String> taxonIds, int alignmentIdentity, double clockRate) {
            this.taxonIds = List.copyOf(taxonIds);
            this.alignmentIdentity = alignmentIdentity;
            this.clockRate = clockRate;
        }
    }

    private record GridData(
            double maxDistance,
            double gridSpacing,
            double[] grid,
            double[][] logLikelihoodGrid,
            double[][] weightGrid,
            double[][] cdf1GivenD0Grid,
            double[] marginal0,
            double[] cdf0,
            double maxLogLikelihood,
            double totalMass
    ) {
    }

}
