package upgma;

import java.util.SplittableRandom;

/**
 * Draws a distance matrix D that is consistent with a fixed tree, following the backward
 * slack-sampling construction in {@code approach.md}.
 *
 * <p>Walking the tree's merges backwards (see {@link UpgmaSplitModel}), every split of a cluster X
 * into its children A and B draws, for every other cluster V still active at that point, how much
 * of the accumulated slack between V and X falls to A and how much to B. Once every internal node
 * has been split back down to its leaves, all pairwise slacks are determined and the distance
 * matrix follows from {@code d(i, j) = d(1) + s_1(i, j)}.
 *
 * <p>The matrix is indexed by tree leaf node number, matching {@code adapters.TreeUtils} and
 * {@code distancematrix.SingleLinkageTree} elsewhere in this codebase, so it can be handed straight
 * to {@code TreeUtils.applyClustering} without remapping.
 */
public class UpgmaDistanceSampler {

    private final UpgmaSplitModel model;
    private final SplittableRandom rng;

    /**
     * @param rng private generator for the Beta draws; the operator seeds it once from {@code Randomizer}
     *            so runs stay reproducible from the BEAST seed without paying the per-call lock
     */
    public UpgmaDistanceSampler(UpgmaSplitModel model, SplittableRandom rng) {
        this.model = model;
        this.rng = rng;
    }

    /**
     * Draws slacks backwards from the root and writes the resulting distance matrix into
     * {@code distances}, indexed by tree leaf node number.
     */
    public void sample(double[][] distances) {
        int n = this.model.leafCount();
        for (int i = 0; i < n; i++) {
            distances[i][i] = 0.0;
        }
        if (this.model.mergeCount() == 0) {
            return;
        }

        // slack[a][b], for a < b, holds s_{r+1}(a, b) for the clusters active at the current backward
        // step, indexed by tree node number. The root's children start out with a slack of 0.
        double[][] slack = new double[2 * n - 1][2 * n - 1];

        for (UpgmaSplitModel.Split split : this.model.splits()) {
            double delta = this.model.mergeDistance(split.r() + 1) - this.model.mergeDistance(split.r());

            // unaffected pairs: s_r(U, V) = s_{r+1}(U, V) + delta
            for (int i = 0; i < split.active().length; i++) {
                for (int j = i + 1; j < split.active().length; j++) {
                    int u = split.active()[i];
                    int v = split.active()[j];
                    setSlack(slack, u, v, getSlack(slack, u, v) + delta);
                }
            }

            // affected pairs: split every other active cluster's slack against X between A and B
            for (int v : split.active()) {
                double base = split.size() * (getSlack(slack, v, split.nodeNr()) + delta);

                double mean = this.model.betaMean(v, split);
                double p = this.sampleBeta(this.model.alpha(mean), this.model.beta(mean));

                setSlack(slack, v, split.leftNr(), p * base / split.leftSize());
                setSlack(slack, v, split.rightNr(), (1.0 - p) * base / split.rightSize());
            }

            setSlack(slack, split.leftNr(), split.rightNr(), 0.0);
        }

        double base = this.model.mergeDistance(1);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = base + getSlack(slack, i, j);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }
    }

    /** Draws Beta(alpha, beta) as X / (X + Y) with X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1). */
    private double sampleBeta(double alpha, double beta) {
        double x = sampleGamma(alpha, this.rng);
        double y = sampleGamma(beta, this.rng);
        return x / (x + y);
    }

    /**
     * Draws Gamma(shape, 1) with the Marsaglia-Tsang (2000) squeeze method: one normal and one uniform
     * per attempt, accepted almost always, with a log only on the rare fallback.
     */
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

    private static double getSlack(double[][] slack, int a, int b) {
        return a < b ? slack[a][b] : slack[b][a];
    }

    private static void setSlack(double[][] slack, int a, int b, double value) {
        if (a < b) {
            slack[a][b] = value;
        } else {
            slack[b][a] = value;
        }
    }
}
