package upgma;

import java.util.SplittableRandom;

/**
 * Draws a distance matrix D consistent with a fixed tree, following the backward slack sampling in
 * {@code approach.md}.
 *
 * <p>Every split of a cluster X into A and B draws how much of each other active cluster's slack
 * against X falls to A and how much to B. Once every internal node is split back down to its
 * leaves, every pairwise distance is determined.
 *
 * <p>Slacks are held as the cluster distances they stand for rather than as slacks. Since
 * {@code s_r(U, V) = d(U, V) - d(r)}, the two carry the same information, but a pair the split does
 * not touch keeps the distance it already had, where its slack would have had to be walked forward
 * by {@code delta_r} at every remaining step. That turns the whole backward pass into the mirror
 * image of {@link UpgmaClustering}: the same dense working matrix over active clusters, splitting
 * where the clustering merges, with A keeping X's slot and B taking a new one at the end.
 *
 * <p>The matrix is indexed by tree leaf node number, so it can go straight to
 * {@code TreeUtils.applyClustering}.
 */
public class UpgmaDistanceSampler {

    private final UpgmaSplitModel model;
    private final SplittableRandom rng;

    /**
     * @param rng private generator for the Beta draws, seeded once from {@code Randomizer} so runs
     *            stay reproducible without paying its per-call lock
     */
    public UpgmaDistanceSampler(UpgmaSplitModel model, SplittableRandom rng) {
        this.model = model;
        this.rng = rng;
    }

    /** Draws the distances backwards from the root and writes them into {@code distances}. */
    public void sample(double[][] distances) {
        int n = this.model.leafCount();
        for (int i = 0; i < n; i++) {
            distances[i][i] = 0.0;
        }
        if (this.model.mergeCount() == 0) {
            return;
        }

        // dense working matrix over the active clusters, slot-indexed like UpgmaClustering's
        double[] d = new double[n * n];
        int[] slotOf = new int[2 * n - 1];
        int[] clusterOf = new int[n];

        // the root's children are the only clusters to start with, and d(n - 1) apart
        int[] initial = this.model.initialClusters();
        for (int slot = 0; slot < 2; slot++) {
            slotOf[initial[slot]] = slot;
            clusterOf[slot] = initial[slot];
        }
        int activeCount = 2;
        double rootDistance = this.model.mergeDistance(this.model.mergeCount());
        d[0 * n + 1] = rootDistance;
        d[1 * n + 0] = rootDistance;

        for (UpgmaSplitModel.Split split : this.model.splits()) {
            double dr = this.model.mergeDistance(split.r());
            int slotA = slotOf[split.nodeNr()];
            int slotB = activeCount++;

            int[] active = split.active();
            for (int i = 0; i < active.length; i++) {
                int slotV = slotOf[active[i]];

                // s_{r+1}(V, X) + delta_r = d(V, X) - d(r), so the slack to divide is read directly
                double base = split.size() * (d[slotV * n + slotA] - dr);

                double mean = this.model.betaMean(split, i);
                double p = this.sampleBeta(this.model.alpha(mean), this.model.beta(mean));

                double dVA = dr + p * base / split.leftSize();
                double dVB = dr + (1.0 - p) * base / split.rightSize();

                // A takes X's slot, so writing d(V, A) overwrites the d(V, X) just read
                d[slotV * n + slotA] = dVA;
                d[slotA * n + slotV] = dVA;
                d[slotV * n + slotB] = dVB;
                d[slotB * n + slotV] = dVB;
            }

            // A and B are what merged at r, so they sit exactly d(r) apart
            d[slotA * n + slotB] = dr;
            d[slotB * n + slotA] = dr;

            slotOf[split.leftNr()] = slotA;
            slotOf[split.rightNr()] = slotB;
            clusterOf[slotA] = split.leftNr();
            clusterOf[slotB] = split.rightNr();
        }

        // every cluster is a single leaf by now, so the working matrix is D up to the slot ordering
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double value = d[i * n + j];
                distances[clusterOf[i]][clusterOf[j]] = value;
                distances[clusterOf[j]][clusterOf[i]] = value;
            }
        }
    }

    /** Draws Beta(alpha, beta) as X / (X + Y) with X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1). */
    private double sampleBeta(double alpha, double beta) {
        double x = sampleGamma(alpha, this.rng);
        double y = sampleGamma(beta, this.rng);
        return x / (x + y);
    }

    /** Draws Gamma(shape, 1) with the Marsaglia-Tsang (2000) squeeze method. */
    private static double sampleGamma(double shape, SplittableRandom rng) {
        if (shape < 1.0) {
            return sampleGamma(shape + 1.0, rng) * Math.pow(rng.nextDouble(), 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        while (true) {
            double x = rng.nextGaussian();
            double v = 1.0 + c * x;
            if (v <= 0.0) {
                continue;
            }
            v = v * v * v;

            double u = rng.nextDouble();
            double x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2 || Math.log(u) < 0.5 * x2 + d * (1.0 - v + Math.log(v))) {
                return d * v;
            }
        }
    }
}
