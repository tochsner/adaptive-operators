package upgma;

import beast.base.util.GammaFunction;

/**
 * Evaluates {@code log p(D | T)} for a distance matrix D already known to satisfy
 * {@code UPGMA(D) = T}, following the same backward construction as {@link UpgmaDistanceSampler}
 * but reading every quantity off D instead of sampling it.
 *
 * <p>For every split of X into A and B and every active cluster V, the fraction the sampler would
 * have drawn is recovered as {@code p_r(V, A) = |A| (d(V, A) - d(r)) / (|X| (d(V, X) - d(r)))} and
 * scored under the Beta shape the sampler would have used, together with the Jacobian factor
 * {@code |X|^2 (d(V, X) - d(r)) / (|A| |B|)} from approach.md.
 */
public class UpgmaConditionalDensity {

    private final UpgmaSplitModel model;

    public UpgmaConditionalDensity(UpgmaSplitModel model) {
        this.model = model;
    }

    /** Returns the Beta log densities of the fractions D implies, minus the log Jacobian. */
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

    /** Average distance in D between two clades. */
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
