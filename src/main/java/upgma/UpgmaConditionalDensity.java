package upgma;

import beast.base.util.GammaFunction;

import java.util.List;

/**
 * Evaluates {@code log p(D | T)} for a distance matrix D already known to satisfy
 * {@code UPGMA(D) = T}, scoring the same splits {@link UpgmaDistanceSampler} draws but reading
 * every quantity off D instead of sampling it.
 *
 * <p>For every split of X into A and B and every active cluster V, the fraction the sampler would
 * have drawn is recovered as {@code p_r(V, A) = |A| (d(V, A) - d(r)) / (|X| (d(V, X) - d(r)))} and
 * scored under the Beta shape the sampler would have used, together with the Jacobian factor
 * {@code |X|^2 (d(V, X) - d(r)) / (|A| |B|)} from approach.md.
 *
 * <p>Where the sampler runs the merges backwards, this walks them forwards, over a dense working
 * copy of D that each merge is applied to as it is scored. Every cluster distance a split needs is
 * then already in that matrix, and {@code d(V, X)} is the merge's own weighted average, so nothing
 * has to be re-averaged over the leaves of the two clades.
 */
public class UpgmaConditionalDensity {

    private final UpgmaSplitModel model;

    public UpgmaConditionalDensity(UpgmaSplitModel model) {
        this.model = model;
    }

    /** Returns the Beta log densities of the fractions D implies, minus the log Jacobian. */
    public double logDensity(double[][] distances) {
        int n = this.model.leafCount();
        List<UpgmaSplitModel.Split> splits = this.model.splits();
        if (splits.isEmpty()) {
            return 0.0;
        }

        // dense working matrix over the active clusters, slot-indexed like UpgmaClustering's
        double[] d = new double[n * n];
        int[] slotOf = new int[2 * n - 1];
        int[] clusterOf = new int[n];
        for (int leaf = 0; leaf < n; leaf++) {
            slotOf[leaf] = leaf;
            clusterOf[leaf] = leaf;
            System.arraycopy(distances[leaf], 0, d, leaf * n, n);
        }

        double logDensity = 0.0;

        // the splits run from the root down, so the merges are that list read backwards
        int activeCount = n;
        for (int index = splits.size() - 1; index >= 0; index--) {
            UpgmaSplitModel.Split split = splits.get(index);
            double dr = this.model.mergeDistance(split.r());
            int slotA = slotOf[split.leftNr()];
            int slotB = slotOf[split.rightNr()];

            int[] active = split.active();
            for (int i = 0; i < active.length; i++) {
                int slotV = slotOf[active[i]];
                double dVA = d[slotV * n + slotA];
                double dVB = d[slotV * n + slotB];
                double dVX = (split.leftSize() * dVA + split.rightSize() * dVB) / split.size();

                double base = split.size() * (dVX - dr);
                double p = split.leftSize() * (dVA - dr) / base;

                double mean = this.model.betaMean(split, i);
                logDensity += logBetaPdf(p, this.model.alpha(mean), this.model.beta(mean));

                // Jacobian of the map from the fractions to D
                logDensity -= Math.log(split.size()) + Math.log(base)
                        - Math.log(split.leftSize()) - Math.log(split.rightSize());

                // d(V, X) is the merge's own weighted average, so folding B into A's slot is free
                d[slotV * n + slotA] = dVX;
                d[slotA * n + slotV] = dVX;
            }
            slotOf[split.nodeNr()] = slotA;
            clusterOf[slotA] = split.nodeNr();

            // compact away B's now-stale slot by moving the last active slot into it
            int last = activeCount - 1;
            if (slotB != last) {
                for (int k = 0; k < last; k++) {
                    d[slotB * n + k] = d[last * n + k];
                    d[k * n + slotB] = d[k * n + last];
                }
                slotOf[clusterOf[last]] = slotB;
                clusterOf[slotB] = clusterOf[last];
            }
            activeCount--;
        }

        return logDensity;
    }

    private static double logBetaPdf(double p, double alpha, double beta) {
        double logNormalizer =
                GammaFunction.lnGamma(alpha) + GammaFunction.lnGamma(beta) - GammaFunction.lnGamma(alpha + beta);
        return (alpha - 1.0) * Math.log(p) + (beta - 1.0) * Math.log(1.0 - p) - logNormalizer;
    }
}
