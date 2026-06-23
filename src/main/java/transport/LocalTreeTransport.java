package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.sitemodel.SiteModelInterface;
import beast.base.spec.inference.parameter.RealScalarParam;
import org.apache.commons.statistics.distribution.NormalDistribution;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LocalTreeTransport {

    private static final int GRID_SIZE = 25;
    private static final int LAST_GRID_INDEX = GRID_SIZE - 1;
    private static final double MIN_GRID_DISTANCE_FRACTION = 1.0 / (2.0 * GRID_SIZE);
    private static final double MIN_CDF_PROBABILITY = 1.0E-15;
    private static final double MAX_CDF_PROBABILITY = 1.0 - MIN_CDF_PROBABILITY;
    private static final double MIN_LOG_DENSITY = 1.0E-300;
    private static final NormalDistribution STANDARD_NORMAL = NormalDistribution.of(0.0, 1.0);

    private final double maxDistance;
    private final double gridSpacing;
    private final double[] grid;
    private final double[] distanceOffsets;

    private final double[][] logLikelihoodGrid;
    private final double[][] cdf1GivenD0Grid;
    private final double[] cdf0;
    private final double maxLogLikelihood;
    private final double totalMass;

    public LocalTreeTransport(List<String> taxonIds, List<Double> taxonHeights, Alignment alignment, SiteModelInterface siteModel, RealScalarParam<?> clockRate, double maxDistance) {
        if (!Double.isFinite(maxDistance) || maxDistance <= 0.0) {
            throw new IllegalArgumentException("maxDistance must be finite and positive");
        }

        GridData gridData = buildGridData(taxonIds, taxonHeights, siteModel, alignment, clockRate, maxDistance);

        this.maxDistance = gridData.maxDistance;
        this.gridSpacing = gridData.gridSpacing;
        this.grid = gridData.grid;
        this.distanceOffsets = gridData.distanceOffsets;
        this.logLikelihoodGrid = gridData.logLikelihoodGrid;
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
        double[] shiftedDistances = this.shiftDistances(distances);
        transported[0] = this.invGaussianCDF(this.felsensteinCDF(0, shiftedDistances));
        transported[1] = this.invGaussianCDF(this.felsensteinCDF(1, shiftedDistances));

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

        return this.unshiftDistances(distances);
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

    public double felsensteinLogPDF(double[] distances) {
        this.validateDistanceDimension(distances);

        double[] shiftedDistances = this.shiftDistances(distances);
        double clamped0 = this.clampDistance(shiftedDistances[0]);
        double clamped1 = this.clampDistance(shiftedDistances[1]);
        return this.bilinearInterpolateLogLikelihood(clamped0, clamped1);
    }

    private static GridData buildGridData(
            List<String> taxonIds,
            List<Double> taxonHeights,
            SiteModelInterface siteModel,
            Alignment alignment,
            RealScalarParam<?> clockRate,
            double maxDistance
    ) {
        double[] distanceOffsets = buildDistanceOffsets(taxonHeights);
        Approximate3TaxaFelsenstein approximateFelsenstein = new Approximate3TaxaFelsenstein(taxonIds, alignment, siteModel, clockRate);

        double minGridDistance = maxDistance * MIN_GRID_DISTANCE_FRACTION;
        double gridSpacing = (maxDistance - minGridDistance) / LAST_GRID_INDEX;
        double[] grid = buildGrid(minGridDistance, gridSpacing);

        double[][] logLikelihoodGrid = new double[GRID_SIZE][GRID_SIZE];
        double maxLogLikelihood = populateLogLikelihoodGrid(approximateFelsenstein, grid, distanceOffsets, logLikelihoodGrid);

        double[][] offsetWeightGrid = buildOffsetWeightGrid(logLikelihoodGrid, maxLogLikelihood);
        double[][] cdf1GivenD0Grid = buildConditionalCDF1Grid(offsetWeightGrid, grid[0], gridSpacing);
        double[] marginal0 = buildMarginal0(cdf1GivenD0Grid);
        double[] cdf0 = buildCDF0(marginal0, grid[0], gridSpacing);
        double totalMass = cdf0[LAST_GRID_INDEX];

        return new GridData(
                maxDistance,
                gridSpacing,
                grid,
                distanceOffsets,
                logLikelihoodGrid,
                cdf1GivenD0Grid,
                cdf0,
                maxLogLikelihood,
                totalMass
        );
    }

    private static double[] buildGrid(double minGridDistance, double gridSpacing) {
        double[] grid = new double[GRID_SIZE];
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i] = minGridDistance + i * gridSpacing;
        }
        return grid;
    }

    private static double[] buildDistanceOffsets(List<Double> taxonHeights) {
        if (taxonHeights.size() != 3) {
            throw new IllegalArgumentException("three-taxon transport requires exactly three taxon heights");
        }

        double[] offsets = new double[2];
        for (int i = 0; i < offsets.length; i++) {
            double firstHeight = taxonHeights.get(i);
            double secondHeight = taxonHeights.get(i + 1);

            if (!Double.isFinite(firstHeight) || !Double.isFinite(secondHeight)) {
                throw new IllegalArgumentException("taxon heights must be finite");
            }

            offsets[i] = Math.abs(firstHeight - secondHeight);
        }
        return offsets;
    }

    private static double populateLogLikelihoodGrid(
            Approximate3TaxaFelsenstein approximateFelsenstein,
            double[] grid,
            double[] distanceOffsets,
            double[][] logLikelihoodGrid
    ) {
        double maxLogLikelihood = Double.NEGATIVE_INFINITY;
        double[] distances = new double[2];

        for (int i = 0; i < GRID_SIZE; i++) {
            distances[0] = grid[i] + distanceOffsets[0];
            for (int j = 0; j < GRID_SIZE; j++) {
                distances[1] = grid[j] + distanceOffsets[1];

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

    private static double[][] buildOffsetWeightGrid(double[][] logLikelihoodGrid, double maxLogLikelihood) {
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

    private static double[][] buildConditionalCDF1Grid(
            double[][] offsetWeightGrid,
            double firstGridDistance,
            double gridSpacing
    ) {
        double[][] cdf = new double[GRID_SIZE][GRID_SIZE];

        for (int i = 0; i < GRID_SIZE; i++) {
            cdf[i][0] = trapezoidArea(0.0, offsetWeightGrid[i][0], firstGridDistance);
            for (int j = 1; j < GRID_SIZE; j++) {
                cdf[i][j] = cdf[i][j - 1] + trapezoidArea(offsetWeightGrid[i][j - 1], offsetWeightGrid[i][j], gridSpacing);
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

    private static double[] buildCDF0(double[] marginal0, double firstGridDistance, double gridSpacing) {
        double[] cdf = new double[GRID_SIZE];
        cdf[0] = trapezoidArea(0.0, marginal0[0], firstGridDistance);
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

    private double bilinearInterpolateLogLikelihood(double x, double y) {
        GridPosition xPosition = this.getGridPosition(x);
        GridPosition yPosition = this.getGridPosition(y);

        double lowerY = this.interpolate(
                this.getFiniteLogLikelihood(xPosition.lowerIndex, yPosition.lowerIndex),
                this.getFiniteLogLikelihood(xPosition.lowerIndex, yPosition.upperIndex),
                yPosition.fraction
        );
        double upperY = this.interpolate(
                this.getFiniteLogLikelihood(xPosition.upperIndex, yPosition.lowerIndex),
                this.getFiniteLogLikelihood(xPosition.upperIndex, yPosition.upperIndex),
                yPosition.fraction
        );
        return this.interpolate(lowerY, upperY, xPosition.fraction);
    }

    private double getFiniteLogLikelihood(int i, int j) {
        double logLikelihood = this.logLikelihoodGrid[i][j];
        if (Double.isFinite(logLikelihood)) {
            return logLikelihood;
        }
        return this.maxLogLikelihood + Math.log(MIN_LOG_DENSITY);
    }

    private double linearInterpolate(double[] xValues, double[] yValues, double x) {
        if (x <= xValues[0]) {
            if (xValues[0] <= 0.0) {
                return yValues[0];
            }
            return this.interpolate(0.0, yValues[0], Math.max(0.0, x / xValues[0]));
        }

        GridPosition position = this.getGridPosition(x);
        return this.interpolate(yValues[position.lowerIndex], yValues[position.upperIndex], position.fraction);
    }

    private double inverseMonotone(double[] xValues, double[] yValues, double y) {
        if (y <= yValues[0]) {
            if (yValues[0] <= 0.0) {
                return 0.0;
            }
            return this.interpolate(0.0, xValues[0], Math.max(0.0, y / yValues[0]));
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
        int lowerIndex = (int) Math.floor((clamped - this.grid[0]) / this.gridSpacing);

        if (lowerIndex >= LAST_GRID_INDEX) {
            return new GridPosition(LAST_GRID_INDEX, LAST_GRID_INDEX, 0.0);
        } else if (lowerIndex < 0) {
            return new GridPosition(0, 0, 0.0);
        }

        int upperIndex = lowerIndex + 1;
        double fraction = (clamped - this.grid[lowerIndex]) / this.gridSpacing;
        return new GridPosition(lowerIndex, upperIndex, fraction);
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

    private double[] shiftDistances(double[] distances) {
        this.validateDistanceDimension(distances);

        double[] shiftedDistances = new double[distances.length];
        for (int i = 0; i < distances.length; i++) {
            shiftedDistances[i] = distances[i] - this.distanceOffsets[i];
            if (!Double.isFinite(shiftedDistances[i]) || shiftedDistances[i] < 0.0) {
                throw new IllegalArgumentException("distance must be finite and at least its taxon-height offset");
            }
        }
        return shiftedDistances;
    }

    public double[] unshiftDistances(double[] shiftedDistances) {
        this.validateDistanceDimension(shiftedDistances);

        double[] distances = new double[shiftedDistances.length];
        for (int i = 0; i < shiftedDistances.length; i++) {
            distances[i] = shiftedDistances[i] + this.distanceOffsets[i];
        }
        return distances;
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

    private record GridData(
            double maxDistance,
            double gridSpacing,
            double[] grid,
            double[] distanceOffsets,
            double[][] logLikelihoodGrid,
            double[][] cdf1GivenD0Grid,
            double[] cdf0,
            double maxLogLikelihood,
            double totalMass
    ) {
    }

}
