package adaptiveoperators;

import java.util.Random;

/**
 * Models the distribution of pairwise taxon distances for a fixed collection of taxon
 * pairs. Each round the operator chooses one pair and asks the model to sample a new
 * distance for it and to evaluate the log density of a candidate distance. The current
 * distances of all pairs are provided so that models conditioning on the other pairs
 * (such as a neural model) have access to them; models that treat pairs independently
 * may ignore everything but the chosen pair.
 */
public interface TaxaDistanceModel {

    /**
     * Records an observation of the current distances. {@code pairIndex} identifies the
     * pair the operator selected this round.
     */
    void record(double[] distances, int pairIndex);

    /**
     * Samples a new distance for the chosen pair conditioned on the current distances.
     */
    double sample(double[] distances, int pairIndex, Random random, double scaleFactor);

    /**
     * Log density of {@code value} as a distance for the chosen pair conditioned on the
     * current distances.
     */
    double logDensity(double[] distances, int pairIndex, double value, double scaleFactor);
}
