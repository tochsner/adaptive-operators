package transport;

import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math4.legacy.analysis.polynomials.PolynomialSplineFunction;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;

public class UnivariateOptimalTransportMap {

    private static final int GRID_SIZE = 12;
    private static final double MIN_DENSITY = 1.0E-300;
    private static final double EDGE_FRACTION = 1.0E-6;
    private static final double MIN_OTHER_SEGMENT_MASS = 1.0E-12;

    private final double min;
    private final double max;
    private final double supportMin;
    private final double supportMax;
    private final PolynomialSplineFunction shiftedLogDensitySpline;
    private final double[] grid;
    private final double[] shiftedLogDensities;
    private final double[] cumulativeMass;
    private final double totalMass;

    public UnivariateOptimalTransportMap(double min, double max, Set<Double> pathHeights, Function<Double, Double> getLogDensity) {
        if (!Double.isFinite(max) || max <= min) {
            throw new IllegalArgumentException("max must be finite and greater than min");
        }

        this.min = min;
        this.max = max;
        double edgeOffset = Math.max(Math.ulp(this.max), (this.max - this.min) * EDGE_FRACTION);
        this.supportMin = this.min + edgeOffset;
        this.supportMax = this.max - edgeOffset;
        if (this.supportMin >= this.supportMax) {
            throw new IllegalArgumentException("transport interval is too small");
        }

        this.grid = getChebyshevGrid(this.supportMin, this.supportMax, pathHeights);
        double[] logDensities = new double[this.grid.length];
        double maxLogDensity = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < this.grid.length; i++) {
            logDensities[i] = getLogDensity.apply(this.grid[i]);

            if (!Double.isFinite(logDensities[i])) {
                logDensities[i] = Math.log(MIN_DENSITY);
            }

            maxLogDensity = Math.max(maxLogDensity, logDensities[i]);
        }

        if (!Double.isFinite(maxLogDensity)) {
            throw new IllegalArgumentException("log density must be finite for at least one grid point");
        }

        this.shiftedLogDensities = getShiftedLogDensities(logDensities, maxLogDensity);
        this.shiftedLogDensitySpline = new LinearInterpolator().interpolate(this.grid, this.shiftedLogDensities);
        this.cumulativeMass = getCumulativeMass(this.grid, this.shiftedLogDensities);
        this.totalMass = this.cumulativeMass[this.cumulativeMass.length - 1];

        if (!Double.isFinite(this.totalMass) || this.totalMass <= 0.0) {
            throw new IllegalArgumentException("normalizing constant must be finite and positive");
        }
    }

    public double sample() {
        return this.quantile(Randomizer.nextDouble());
    }

    /**
     * Samples from the map restricted to everything outside the segment holding {@code current},
     * where the segments are the pieces of the support cut at {@code boundaries}. Used to propose
     * only attachments on another branch (a Metropolized Gibbs step over branches).
     *
     * <p>The forward density is m(y) / (1 - M(seg(x))) for y outside seg(x), so the log Hastings
     * ratio is [log m(x) - log(1 - M(seg(y)))] - [log m(y) - log(1 - M(seg(x)))].
     */
    double[] sampleExcludingSegment(double current, double[] boundaries) {
        double[] currentSegment = this.segment(current, boundaries);
        double massBelowCurrent = this.massFractionBelow(currentSegment[0]);
        double currentMass = this.massFractionBelow(currentSegment[1]) - massBelowCurrent;
        double otherMass = 1.0 - currentMass;
        if (otherMass < MIN_OTHER_SEGMENT_MASS) {
            return null;
        }

        // map a uniform draw over the other segments' mass around the excluded segment
        double u = Randomizer.nextDouble() * otherMass;
        double proposed = this.quantile(u < massBelowCurrent ? u : u + currentMass);

        double[] proposedSegment = this.segment(proposed, boundaries);
        if (proposedSegment[0] == currentSegment[0]) {
            // only reachable through rounding at a segment boundary
            return null;
        }

        double proposedMass = this.massFractionBelow(proposedSegment[1]) - this.massFractionBelow(proposedSegment[0]);
        double reverseOtherMass = 1.0 - proposedMass;
        if (reverseOtherMass < MIN_OTHER_SEGMENT_MASS) {
            return null;
        }

        double logHR = (this.logDensity(current) - Math.log(reverseOtherMass))
                - (this.logDensity(proposed) - Math.log(otherMass));
        return new double[] {proposed, logHR};
    }

    /** Returns [lower, upper) of the segment holding {@code value}. */
    private double[] segment(double value, double[] boundaries) {
        double lower = this.supportMin;
        double upper = this.supportMax;

        for (double boundary : boundaries) {
            if (boundary <= value) {
                lower = Math.max(lower, boundary);
            } else {
                upper = Math.min(upper, boundary);
            }
        }

        return new double[] {lower, upper};
    }

    /** Inverse of the map's cumulative distribution. */
    double quantile(double p) {
        double targetMass = p * this.totalMass;
        int interval = getIntervalForMass(this.cumulativeMass, targetMass);
        double localMass = targetMass - this.cumulativeMass[interval];

        return invertIntervalMass(
                this.grid[interval],
                this.grid[interval + 1],
                this.shiftedLogDensities[interval],
                this.shiftedLogDensities[interval + 1],
                localMass
        );
    }

    public double logHRCorrection(double oldValue, double newValue) {
        if (!this.isInsideSamplingSupport(oldValue) || !this.isInsideSamplingSupport(newValue)) {
            return Double.NEGATIVE_INFINITY;
        }

        return this.logDensity(oldValue) - this.logDensity(newValue);
    }

    public boolean isInsideSamplingSupport(double value) {
        return value >= this.supportMin && value <= this.supportMax;
    }

    public double logDensity(double value) {
        if (!this.isInsideSamplingSupport(value)) {
            throw new IllegalArgumentException("value must be inside the sampling support");
        }

        return this.shiftedLogDensitySpline.value(value) - Math.log(this.totalMass);
    }

    /** Fraction of the total mass that lies below {@code value}. */
    double massFractionBelow(double value) {
        if (value <= this.grid[0]) {
            return 0.0;
        }

        if (value >= this.grid[this.grid.length - 1]) {
            return 1.0;
        }

        int interval = Arrays.binarySearch(this.grid, value);
        if (interval < 0) {
            interval = -interval - 2;
        }
        interval = Math.min(interval, this.grid.length - 2);

        double width = this.grid[interval + 1] - this.grid[interval];
        double slope = this.shiftedLogDensities[interval + 1] - this.shiftedLogDensities[interval];
        double partialWidth = value - this.grid[interval];
        double partialMass = getIntervalMass(
                this.grid[interval],
                value,
                this.shiftedLogDensities[interval],
                this.shiftedLogDensities[interval] + slope * partialWidth / width
        );

        return (this.cumulativeMass[interval] + partialMass) / this.totalMass;
    }

    private static double[] getShiftedLogDensities(double[] logDensities, double shift) {
        double[] shiftedLogDensities = new double[logDensities.length];

        for (int i = 0; i < logDensities.length; i++) {
            shiftedLogDensities[i] = Math.max(Math.log(MIN_DENSITY), logDensities[i] - shift);
        }

        return shiftedLogDensities;
    }

    private static double[] getCumulativeMass(double[] grid, double[] shiftedLogDensities) {
        double[] cumulativeMass = new double[grid.length];

        for (int i = 0; i < grid.length - 1; i++) {
            cumulativeMass[i + 1] = cumulativeMass[i] + getIntervalMass(
                    grid[i],
                    grid[i + 1],
                    shiftedLogDensities[i],
                    shiftedLogDensities[i + 1]
            );
        }

        return cumulativeMass;
    }

    private static double getIntervalMass(double left, double right, double leftLogDensity, double rightLogDensity) {
        double width = right - left;
        double slope = rightLogDensity - leftLogDensity;

        if (Math.abs(slope) < 1.0E-12) {
            return width * Math.exp(leftLogDensity);
        }

        return width * Math.exp(leftLogDensity) * Math.expm1(slope) / slope;
    }

    private static double invertIntervalMass(
            double left,
            double right,
            double leftLogDensity,
            double rightLogDensity,
            double mass
    ) {
        double width = right - left;
        double slope = rightLogDensity - leftLogDensity;
        double leftDensity = Math.exp(leftLogDensity);

        if (Math.abs(slope) < 1.0E-12) {
            return left + mass / leftDensity;
        }

        double scaledMass = mass * slope / (width * leftDensity);
        double fraction = Math.log1p(scaledMass) / slope;
        return left + width * Math.min(1.0, Math.max(0.0, fraction));
    }

    private static int getIntervalForMass(double[] cumulativeMass, double targetMass) {
        int index = Arrays.binarySearch(cumulativeMass, targetMass);
        if (index >= 0) {
            return Math.min(index, cumulativeMass.length - 2);
        }

        return Math.max(0, Math.min(-index - 2, cumulativeMass.length - 2));
    }

    private static double[] getChebyshevGrid(double min, double max, Set<Double> additionalHeights) {
        Set<Double> grid = new HashSet<>();

        double midpoint = 0.5 * (min + max);
        double halfWidth = 0.5 * (max - min);

        int additionalHeightCount = 0;
        for (double height : additionalHeights) {
            if (height > min && height < max) {
                additionalHeightCount++;
            }
        }

        int gridSize = Math.max(2, GRID_SIZE - additionalHeightCount);

        for (int i = 0; i < gridSize; i++) {
            int reverseIndex = gridSize - 1 - i;
            double angle = reverseIndex * Math.PI / (gridSize - 1);
            grid.add(midpoint + halfWidth * Math.cos(angle));
        }

        for (double height : additionalHeights) {
            if (height > min && height < max && Randomizer.nextDouble() <= (1.0 * GRID_SIZE / additionalHeightCount)) {
                grid.add(height);
            }
        }

        return grid.stream().sorted(Double::compareTo).mapToDouble(x -> x).toArray();
    }
}
