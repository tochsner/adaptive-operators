package transport;

import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math4.legacy.analysis.polynomials.PolynomialSplineFunction;

import java.util.Arrays;
import java.util.function.Function;

public class UnivariateOptimalTransportMap {

    private static final int GRID_SIZE = 8;
    private static final double MIN_DENSITY = 1.0E-300;
    private static final double EDGE_FRACTION = 1.0E-2;

    private final double min;
    private final double max;
    private final double supportMin;
    private final double supportMax;
    private final double support;
    private final PolynomialSplineFunction logDensitySpline;
    private final double[] grid;
    private final double[] shiftedLogDensities;
    private final double[] cumulativeMass;
    private final double totalMass;

    public UnivariateOptimalTransportMap(double min, double max, Function<Double, Double> getLogDensity) {
        if (!Double.isFinite(max) || max <= min) {
            throw new IllegalArgumentException("max must be finite and greater than min");
        }

        this.min = min;
        this.max = max;
        double edgeOffset = Math.max(Math.ulp(this.max), (this.max - this.min) * EDGE_FRACTION);
        this.supportMin = this.min + edgeOffset;
        this.supportMax = this.max - edgeOffset;
        this.support = this.supportMax - this.supportMin;
        if (this.supportMin >= this.supportMax) {
            throw new IllegalArgumentException("transport interval is too small");
        }

        this.grid = getChebyshevGrid(this.supportMin, this.supportMax);
        double[] logDensities = new double[GRID_SIZE];
        double maxLogDensity = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < GRID_SIZE; i++) {
            logDensities[i] = getLogDensity.apply(this.grid[i]);

            if (!Double.isFinite(logDensities[i])) {
                logDensities[i] = Math.log(MIN_DENSITY);
            }

            maxLogDensity = Math.max(maxLogDensity, logDensities[i]);
        }

        if (!Double.isFinite(maxLogDensity)) {
            throw new IllegalArgumentException("log density must be finite for at least one grid point");
        }

        this.logDensitySpline = new LinearInterpolator().interpolate(this.grid, logDensities);
        this.shiftedLogDensities = getShiftedLogDensities(logDensities, maxLogDensity);
        this.cumulativeMass = getCumulativeMass(this.grid, this.shiftedLogDensities);
        this.totalMass = this.cumulativeMass[this.cumulativeMass.length - 1];

        if (!Double.isFinite(this.totalMass) || this.totalMass <= 0.0) {
            throw new IllegalArgumentException("normalizing constant must be finite and positive");
        }
    }

    public double sample() {
        double targetMass = Randomizer.nextDouble() * this.totalMass;
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
        return this.logDensity(oldValue) - this.logDensity(newValue);
    }

    private double cdf(double value) {
        if (value <= this.supportMin) {
            return 0.0;
        }

        if (value >= this.supportMax) {
            return 1.0;
        }

        int interval = getIntervalForValue(this.grid, value);
        double intervalMass = getIntervalMass(
                this.grid[interval],
                value,
                this.shiftedLogDensities[interval],
                interpolateShiftedLogDensity(interval, value)
        );
        double cumulativeMass = this.cumulativeMass[interval] + intervalMass;

        return Math.min(1.0, Math.max(0.0, cumulativeMass / this.totalMass));
    }

    public double logDensity(double value) {
        if (value < this.min || value > this.max) {
            throw new IllegalArgumentException("value must be inside the transport interval");
        }

        double clampedValue = Math.min(this.supportMax - this.support * 1e-5, Math.max(this.supportMin + this.support * 1e-5, value));
        return this.logDensitySpline.value(clampedValue);
    }

    private double interpolateShiftedLogDensity(int interval, double value) {
        double left = this.grid[interval];
        double right = this.grid[interval + 1];
        double fraction = (value - left) / (right - left);
        return this.shiftedLogDensities[interval]
                + fraction * (this.shiftedLogDensities[interval + 1] - this.shiftedLogDensities[interval]);
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

    private static int getIntervalForValue(double[] grid, double value) {
        int index = Arrays.binarySearch(grid, value);
        if (index >= 0) {
            return Math.min(index, grid.length - 2);
        }

        return Math.max(0, Math.min(-index - 2, grid.length - 2));
    }

    private static double[] getChebyshevGrid(double min, double max) {
        double[] grid = new double[GRID_SIZE];
        double midpoint = 0.5 * (min + max);
        double halfWidth = 0.5 * (max - min);

        for (int i = 0; i < GRID_SIZE; i++) {
            int reverseIndex = GRID_SIZE - 1 - i;
            double angle = reverseIndex * Math.PI / (GRID_SIZE - 1);
            grid[i] = midpoint + halfWidth * Math.cos(angle);
        }

        return grid;
    }
}
