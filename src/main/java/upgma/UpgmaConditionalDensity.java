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
 * has to be re-averaged over the leaves of the two clades. X takes A's slot and B's slot goes dead,
 * since the split lists the clusters still active and nothing ever scans the whole matrix. A density
 * holds its working memory, so it is built once for a leaf count and reused across models.
 */
public class UpgmaConditionalDensity {

    // dense working matrix over the active clusters, slot-indexed like UpgmaClustering's
    private final double[] d;
    private final int[] slotOf;

    public UpgmaConditionalDensity(int leafCount) {
        this.d = new double[leafCount * leafCount];
        this.slotOf = new int[2 * leafCount - 1];
    }

    /** Returns the Beta log densities of the fractions D implies, minus the log Jacobian. */
    public double logDensity(UpgmaSplitModel model, double[][] distances) {
        int n = model.leafCount();
        List<UpgmaSplitModel.Split> splits = model.splits();
        if (splits.isEmpty()) {
            return 0.0;
        }

        double[] d = this.d;
        int[] slotOf = this.slotOf;
        for (int leaf = 0; leaf < n; leaf++) {
            slotOf[leaf] = leaf;
            System.arraycopy(distances[leaf], 0, d, leaf * n, n);
        }

        // the Beta normaliser's lnGamma(alpha + beta) is the same for every fraction, and a good share
        // of the means sit exactly on a clamp, where the other two lnGamma terms are shared as well
        double concentration = model.concentration();
        double logGammaConcentration = GammaFunction.lnGamma(concentration);
        double lowMean = UpgmaSplitModel.MIN_MEAN;
        double highMean = 1.0 - UpgmaSplitModel.MIN_MEAN;
        double lowMeanNormalizer = logGammaShapes(concentration, lowMean);
        double highMeanNormalizer = logGammaShapes(concentration, highMean);

        double logDensity = 0.0;

        // the splits run from the root down, so the merges are that list read backwards
        for (int index = splits.size() - 1; index >= 0; index--) {
            UpgmaSplitModel.Split split = splits.get(index);
            double dr = model.mergeDistance(split.r());
            int slotA = slotOf[split.leftNr()];
            int slotB = slotOf[split.rightNr()];
            int rowA = slotA * n;
            int rowB = slotB * n;
            double leftSize = split.leftSize();
            double rightSize = split.rightSize();
            double size = leftSize + rightSize;

            // with p = |A| (d(V, A) - d(r)) / base and 1 - p = |B| (d(V, B) - d(r)) / base, the Beta
            // log density (alpha - 1) log p + (beta - 1) log(1 - p) less the Jacobian's log(base)
            // rearranges to (alpha - 1) log(d(V, A) - d(r)) + (beta - 1) log(d(V, B) - d(r)) plus
            // per-split constants and (1 - concentration) log(base). The last coefficient is the
            // same for every V, so those logs are taken over a running product instead, one at a
            // time only when the product is about to leave the double range.
            double alphaSum = 0.0;
            double logExcess = 0.0;
            double logBase = 0.0;
            double baseProduct = 1.0;
            int[] active = split.active();
            int offset = split.meanOffset();
            for (int i = 0; i < active.length; i++) {
                int slotV = slotOf[active[i]];
                double dVA = d[rowA + slotV];
                double dVB = d[rowB + slotV];
                double dVX = (leftSize * dVA + rightSize * dVB) / size;

                double mean = model.betaMean(split, i);
                double alpha = concentration * mean;
                double beta = concentration - alpha;
                double normalizer = mean == lowMean ? lowMeanNormalizer
                        : mean == highMean ? highMeanNormalizer : logGammaShapes(concentration, mean);
                alphaSum += alpha;
                logExcess += (alpha - 1.0) * Math.log(dVA - dr) + (beta - 1.0) * Math.log(dVB - dr) - normalizer;

                baseProduct *= size * (dVX - dr);
                if (baseProduct < 1e-150 || baseProduct > 1e150) {
                    logBase += Math.log(baseProduct);
                    baseProduct = 1.0;
                }

                // d(V, X) is the merge's own weighted average, so folding B into A's slot is free
                d[rowA + slotV] = dVX;
                d[slotV * n + slotA] = dVX;
            }
            logBase += Math.log(baseProduct);

            // the per-split constants: sum of (alpha - 1) log |A| + (beta - 1) log |B| over V, the
            // shared normaliser and the rest of the Jacobian, log |A| + log |B| - log |X|
            int count = active.length;
            double betaSum = concentration * count - alphaSum;
            logDensity += logExcess + (1.0 - concentration) * logBase
                    + alphaSum * Math.log(leftSize) + betaSum * Math.log(rightSize)
                    + count * (logGammaConcentration - Math.log(size));

            slotOf[split.nodeNr()] = slotA;
        }

        return logDensity;
    }

    /** {@code lnGamma(alpha) + lnGamma(beta)} for the shapes a mean stands for. */
    private static double logGammaShapes(double concentration, double mean) {
        double alpha = concentration * mean;
        return GammaFunction.lnGamma(alpha) + GammaFunction.lnGamma(concentration - alpha);
    }
}
