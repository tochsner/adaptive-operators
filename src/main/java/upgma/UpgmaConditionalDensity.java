package upgma;

import beast.base.util.GammaFunction;

/**
 * Evaluates the conditional log density {@code log p(D | T)} of a distance matrix D that is already
 * known to satisfy {@code UPGMA(D) = T}, following the same backward construction as
 * {@link UpgmaDistanceSampler} but reading every quantity off D instead of sampling it.
 *
 * <p>For every split of a cluster X into A and B, and every cluster V still active at that point,
 * the fraction {@code p_r(V, A)} that the sampler would have drawn is recovered from D as
 * {@code |A| (d(V, A) - d(r)) / (|X| (d(V, X) - d(r)))}, using {@code s_{r+1}(V, X) + delta_r =
 * d(V, X) - d(r)} from approach.md's slack recursion. Its log density under the Beta shape the
 * sampler would have used is accumulated, together with the matching Jacobian factor
 * {@code |X|^2 (d(V, X) - d(r)) / (|A| |B|)} from approach.md's density formula.
 *
 * <p>Cluster-pair distances are read off D by averaging over the relevant leaf pairs rather than
 * propagated recursively through slacks, since the full matrix is already available. The Beta shapes
 * come from the shared {@link UpgmaSplitModel}, since the Hastings ratio in
 * {@link UpgmaRefreshOperator} depends on sampler and density agreeing exactly.
 */
public class UpgmaConditionalDensity {

    private final UpgmaSplitModel model;

    public UpgmaConditionalDensity(UpgmaSplitModel model) {
        this.model = model;
    }

    /**
     * Returns {@code log p(D | T)}: the sum of the Beta log densities of every fraction D implies,
     * minus the log Jacobian of the map from those fractions to D.
     */
    public double logDensity(double[][] distances) {
        double logDensity = 0.0;

        for (UpgmaSplitModel.Split split : this.model.splits()) {
            double dr = this.model.mergeDistance(split.r());

            for (int v : split.active()) {
                double dVA = this.clusterDistance(distances, v, split.leftNr());
                double dVB = this.clusterDistance(distances, v, split.rightNr());
                double dVX = (split.leftSize() * dVA + split.rightSize() * dVB) / split.size();

                double base = split.size() * (dVX - dr);
                double p = split.leftSize() * (dVA - dr) / base;

                double mean = this.model.betaMean(v, split);
                logDensity += logBetaPdf(p, this.model.alpha(mean), this.model.beta(mean));

                // Jacobian of the map from the fractions to D
                logDensity -= Math.log(split.size()) + Math.log(base)
                        - Math.log(split.leftSize()) - Math.log(split.rightSize());
            }
        }

        return logDensity;
    }

    /** Average distance in D between two clades, i.e. the average-linkage cluster distance. */
    private double clusterDistance(double[][] distances, int nodeV, int nodeA) {
        double sum = 0.0;
        for (int v : this.model.clade(nodeV)) {
            for (int a : this.model.clade(nodeA)) {
                sum += distances[v][a];
            }
        }
        return sum / (this.model.cladeSize(nodeV) * (double) this.model.cladeSize(nodeA));
    }

    private static double logBetaPdf(double p, double alpha, double beta) {
        double logNormalizer =
                GammaFunction.lnGamma(alpha) + GammaFunction.lnGamma(beta) - GammaFunction.lnGamma(alpha + beta);
        return (alpha - 1.0) * Math.log(p) + (beta - 1.0) * Math.log(1.0 - p) - logNormalizer;
    }
}
