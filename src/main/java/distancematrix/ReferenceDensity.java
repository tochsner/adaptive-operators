package distancematrix;

/**
 * Reference density placed independently on every entry of the distance matrix.
 *
 * <p>The matrix carries {@code n(n-1)/2} entries but the tree it induces has only {@code n - 1}
 * heights, so the map from matrices to trees is many-to-one and its fibres are unbounded. Left
 * alone, a random walk on the matrix would drift along those fibres and the tree marginal would come
 * out as {@code pi(T) * V(T)} rather than {@code pi(T)}. Anchoring the entries with a proper
 * reference density makes the fibre volume finite, and single linkage makes it computable in closed
 * form, so it can be divided out exactly. See {@link #logFibreVolume}.
 *
 * <p>Implementations must be supported on {@code (0, inf)}, so that a proposal driving an entry to
 * zero or below is rejected through an infinite log density rather than through a special case.
 */
public interface ReferenceDensity {

    /** Log density at {@code value}, or negative infinity outside the support. */
    double logDensity(double value);

    /** Log of {@code P(X > value)}, or zero for non-positive values. */
    double logSurvival(double value);

    /**
     * Draws from the density conditioned on exceeding {@code threshold}.
     *
     * <p>Used to rebuild the matrix from its exact conditional given a tree, which is what keeps the
     * two in step when something outside this operator changes the tree.
     */
    double sampleAbove(double threshold);

    /**
     * Log fibre volume of the tree induced by the given clustering.
     *
     * <p>Every taxon pair has exactly one most recent common ancestor, so the pairs partition into
     * {@code n - 1} groups, one per internal node, of size {@code m_v = |A_v| * |B_v|}. Holding the
     * topology fixed, the clustering yields a given tree exactly when every entry of a group is at
     * least the group's node distance {@code w_v = 2 * h_v} and the minimum attains it. The fibre is
     * therefore a product over groups, and the induced density of the heights is just the density of
     * {@code n - 1} independent group minima:
     *
     * <pre>
     *     log V(T) = sum_v [ log m_v + log g(w_v) + (m_v - 1) * log Gbar(w_v) ]
     * </pre>
     *
     * Dividing the target by this makes the tree marginal exactly {@code pi(T)}.
     */
    default double logFibreVolume(SingleLinkageTree clustering) {
        double logVolume = 0.0;

        for (int merge = 0; merge < clustering.getLeafCount() - 1; merge++) {
            int crossPairs = clustering.getCrossPairCount(merge);
            double distance = 2.0 * clustering.getMergeHeight(merge);

            logVolume += Math.log(crossPairs)
                    + logDensity(distance)
                    + (crossPairs - 1) * logSurvival(distance);
        }

        return logVolume;
    }
}
