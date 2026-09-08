package upgma;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.distance.Distance;
import beast.base.evolution.tree.Tree;

/**
 * Per-pair Gaussian surrogate of the Jukes-Cantor pairwise likelihood, in substitution units.
 *
 * <p>The mean is the Jukes-Cantor distance, capped like {@code JukesCantorDistance}. The variance
 * is the Laplace approximation {@code p (1 - p) / (L p'(delta)^2)}, read at the capped distance so
 * saturated pairs get a wide surrogate. The mismatch fraction it reads is clamped into
 * {@code [0.5 / L, c]} to keep it positive and finite. Both matrices are indexed by leaf node number.
 */
public final class UpgmaJukesCantorSurrogate {

    private final double[][] distances;
    private final double[][] variances;

    public UpgmaJukesCantorSurrogate(Alignment alignment, Tree tree) {
        int n = tree.getLeafNodeCount();
        this.distances = new double[n][n];
        this.variances = new double[n][n];

        DataType dataType = alignment.getDataType();
        double c = (dataType.getStateCount() - 1.0) / dataType.getStateCount();
        int patternCount = alignment.getPatternCount();

        for (int i = 0; i < n; i++) {
            int taxonI = alignment.getTaxonIndex(tree.getNode(i).getID());
            for (int j = i + 1; j < n; j++) {
                int taxonJ = alignment.getTaxonIndex(tree.getNode(j).getID());

                double mismatches = 0.0;
                double sites = 0.0;
                for (int pattern = 0; pattern < patternCount; pattern++) {
                    int[] states = alignment.getPattern(pattern);
                    int stateI = states[taxonI];
                    int stateJ = states[taxonJ];
                    double weight = alignment.getPatternWeight(pattern);
                    if (!dataType.isAmbiguousCode(stateI) && !dataType.isAmbiguousCode(stateJ) && stateI != stateJ) {
                        mismatches += weight;
                    }
                    sites += weight;
                }

                double p = mismatches / sites;
                double distance = distance(p, c);
                double variance = variance(Math.min(Math.max(p, 0.5 / sites), c), distance, sites, c);
                this.distances[i][j] = distance;
                this.distances[j][i] = distance;
                this.variances[i][j] = variance;
                this.variances[j][i] = variance;
            }
        }
    }

    /** Jukes-Cantor distance for mismatch fraction {@code p}, capped. */
    static double distance(double p, double c) {
        if (p == 0.0) {
            return 0.0;
        }
        if (p >= c) {
            return Distance.Base.MAX_DISTANCE;
        }
        return Math.min(-c * Math.log(1.0 - p / c), Distance.Base.MAX_DISTANCE);
    }

    /** Laplace variance {@code p (1 - p) / (L p'(delta)^2)}. */
    static double variance(double p, double delta, double sites, double c) {
        double slope = Math.exp(-delta / c);
        return p * (1.0 - p) / (sites * slope * slope);
    }

    /** Distances in substitutions per site. */
    public double[][] distances() {
        return this.distances;
    }

    /** Variances in squared substitutions per site. */
    public double[][] variances() {
        return this.variances;
    }
}
