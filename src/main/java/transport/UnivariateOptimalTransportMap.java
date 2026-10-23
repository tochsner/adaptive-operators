package transport;

import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math4.legacy.analysis.polynomials.PolynomialSplineFunction;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;

public class UnivariateOptimalTransportMap {

    private static final int GRID_SIZE = 6;
    private static final int REFINEMENT = 16;
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

        double[] nodes = getChebyshevGrid(this.supportMin, this.supportMax, pathHeights);
        double[] logDensities = new double[nodes.length];
        double maxLogDensity = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < nodes.length; i++) {
            logDensities[i] = getLogDensity.apply(nodes[i]);

            if (!Double.isFinite(logDensities[i])) {
                logDensities[i] = Math.log(MIN_DENSITY);
            }

            maxLogDensity = Math.max(maxLogDensity, logDensities[i]);
        }

        if (!Double.isFinite(maxLogDensity)) {
            throw new IllegalArgumentException("log density must be finite for at least one grid point");
        }

        // interpolate the log density between the nodes with a monotone cubic, tabulated on a fine grid.
        // The proposal is exactly the piecewise exp-linear density on that grid, so sampling and the
        // Hastings ratio stay exact, and the refinement costs no extra density evaluations

        double[] shiftedNodeLogDensities = getShiftedLogDensities(logDensities, maxLogDensity);
        this.grid = getRefinedGrid(nodes);
        this.shiftedLogDensities = getMonotoneCubicValues(nodes, shiftedNodeLogDensities, this.grid);
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

    /** Splits every interval between two nodes into {@code REFINEMENT} equal pieces. */
    private static double[] getRefinedGrid(double[] nodes) {
        double[] refinedGrid = new double[(nodes.length - 1) * REFINEMENT + 1];
        int size = 0;

        for (int i = 0; i < nodes.length - 1; i++) {
            for (int j = 0; j < REFINEMENT; j++) {
                double point = nodes[i] + (nodes[i + 1] - nodes[i]) * j / REFINEMENT;

                // very short intervals can round to repeated points, which the interpolator rejects
                if (size == 0 || point > refinedGrid[size - 1]) {
                    refinedGrid[size++] = point;
                }
            }
        }

        if (nodes[nodes.length - 1] > refinedGrid[size - 1]) {
            refinedGrid[size++] = nodes[nodes.length - 1];
        } else {
            refinedGrid[size - 1] = nodes[nodes.length - 1];
        }

        return Arrays.copyOf(refinedGrid, size);
    }

    /**
     * Evaluates the monotone piecewise cubic Hermite interpolant (PCHIP, Fritsch-Carlson slopes) of
     * the node values at the given points. Unlike a cubic spline it never overshoots the node values,
     * so it cannot put spurious mass next to the sharp peaks and kinks of the log likelihood.
     */
    private static double[] getMonotoneCubicValues(double[] nodes, double[] values, double[] points) {
        int n = nodes.length;
        double[] result = new double[points.length];

        if (n < 3) {
            PolynomialSplineFunction linear = new LinearInterpolator().interpolate(nodes, values);
            for (int k = 0; k < points.length; k++) {
                result[k] = linear.value(points[k]);
            }
            return result;
        }

        double[] widths = new double[n - 1];
        double[] secants = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            widths[i] = nodes[i + 1] - nodes[i];
            secants[i] = (values[i + 1] - values[i]) / widths[i];
        }

        // weighted harmonic mean of the neighbouring secants, zero at local extrema

        double[] slopes = new double[n];
        for (int i = 1; i < n - 1; i++) {
            if (secants[i - 1] * secants[i] > 0) {
                double weight1 = 2 * widths[i] + widths[i - 1];
                double weight2 = widths[i] + 2 * widths[i - 1];
                slopes[i] = (weight1 + weight2) / (weight1 / secants[i - 1] + weight2 / secants[i]);
            }
        }
        slopes[0] = getEndSlope(widths[0], widths[1], secants[0], secants[1]);
        slopes[n - 1] = getEndSlope(widths[n - 2], widths[n - 3], secants[n - 2], secants[n - 3]);

        for (int k = 0; k < points.length; k++) {
            int i = Arrays.binarySearch(nodes, points[k]);
            if (i < 0) {
                i = -i - 2;
            }
            i = Math.max(0, Math.min(i, n - 2));

            double t = (points[k] - nodes[i]) / widths[i];
            double t2 = t * t;
            double t3 = t2 * t;
            double value = (2 * t3 - 3 * t2 + 1) * values[i]
                    + (t3 - 2 * t2 + t) * widths[i] * slopes[i]
                    + (-2 * t3 + 3 * t2) * values[i + 1]
                    + (t3 - t2) * widths[i] * slopes[i + 1];
            result[k] = Math.max(Math.log(MIN_DENSITY), value);
        }

        return result;
    }

    /** Three-point end slope, limited so that the end intervals stay monotone. */
    private static double getEndSlope(double width0, double width1, double secant0, double secant1) {
        double slope = ((2 * width0 + width1) * secant0 - width0 * secant1) / (width0 + width1);

        if (Math.signum(slope) != Math.signum(secant0)) {
            return 0.0;
        }

        if (Math.signum(secant0) != Math.signum(secant1) && Math.abs(slope) > Math.abs(3 * secant0)) {
            return 3 * secant0;
        }

        return slope;
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
