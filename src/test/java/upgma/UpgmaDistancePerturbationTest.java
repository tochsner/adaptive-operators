package upgma;

import beast.base.evolution.tree.Tree;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;
import java.util.SplittableRandom;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * The AR(1) kernel has to report the Hastings correction of the move it actually made: the log
 * ratio of the reverse to the forward Gaussian transition density, summed over the affected pairs,
 * with the adaptive rate held fixed on both sides.
 */
class UpgmaDistancePerturbationTest {

    private static final int LEAF_COUNT = 7;

    @ParameterizedTest
    @ValueSource(doubles = {0.05, 0.3, 0.9, 1.0})
    void ar1CorrectionIsTheTransitionDensityRatio(double beta) {
        Random random = new Random(Double.hashCode(beta));
        Tree tree = UpgmaFixtures.randomTree(LEAF_COUNT, random);

        double[][] substitutionDistances = UpgmaFixtures.randomSymmetric(LEAF_COUNT, random);
        double[][] substitutionVariances = UpgmaFixtures.randomSymmetric(LEAF_COUNT, random);
        double[] tipDates = new double[LEAF_COUNT];
        for (int i = 0; i < LEAF_COUNT; i++) {
            tipDates[i] = 0.1 * random.nextDouble();
        }

        UpgmaTaxonSelection selection = new UpgmaTaxonSelection(UpgmaTaxonSelection.Mode.ALL, tree, 0);
        UpgmaDistancePerturbation perturbation = new UpgmaDistancePerturbation(
                UpgmaDistancePerturbation.Mode.AR1, selection, new SplittableRandom(3L));
        perturbation.setSurrogate(substitutionDistances, substitutionVariances, tipDates);
        perturbation.setScaleFactor(beta);

        double[][] current = UpgmaFixtures.randomSymmetric(LEAF_COUNT, random);
        double[][] proposed = new double[LEAF_COUNT][LEAF_COUNT];
        double logRatio = perturbation.perturb(current, proposed);

        // the first proposal seeds the rate from the matrix it starts from
        double rate = perturbation.getRate();
        assertThat(rate).isCloseTo(referenceRate(substitutionDistances, tipDates, current), within(1e-12));

        double expected = 0.0;
        for (int i = 0; i < LEAF_COUNT; i++) {
            for (int j = i + 1; j < LEAF_COUNT; j++) {
                assertThat(proposed[j][i]).isEqualTo(proposed[i][j]);

                // reverse over forward, which is what the operator adds to the Hastings ratio
                expected += logTransition(current[i][j], proposed[i][j],
                        substitutionDistances[i][j], substitutionVariances[i][j], tipDates[i] + tipDates[j], rate, beta)
                        - logTransition(proposed[i][j], current[i][j],
                        substitutionDistances[i][j], substitutionVariances[i][j], tipDates[i] + tipDates[j], rate, beta);
            }
        }

        assertThat(logRatio).isCloseTo(expected, within(1e-9));
    }

    /** The rate only moves in {@code adaptRate}, towards the running log-mean of the observed rates. */
    @ParameterizedTest
    @ValueSource(longs = {4L, 5L})
    void rateAdaptsToTheRunningLogMeanBetweenProposals(long seed) {
        Random random = new Random(seed);
        Tree tree = UpgmaFixtures.randomTree(LEAF_COUNT, random);

        double[][] substitutionDistances = UpgmaFixtures.randomSymmetric(LEAF_COUNT, random);
        double[] tipDates = new double[LEAF_COUNT];
        for (int i = 0; i < LEAF_COUNT; i++) {
            tipDates[i] = 0.1 * random.nextDouble();
        }

        UpgmaTaxonSelection selection = new UpgmaTaxonSelection(UpgmaTaxonSelection.Mode.ALL, tree, 0);
        UpgmaDistancePerturbation perturbation = new UpgmaDistancePerturbation(
                UpgmaDistancePerturbation.Mode.AR1, selection, new SplittableRandom(seed));
        perturbation.setSurrogate(substitutionDistances, UpgmaFixtures.randomSymmetric(LEAF_COUNT, random), tipDates);
        perturbation.setScaleFactor(0.2);

        double[][] proposed = new double[LEAF_COUNT][LEAF_COUNT];
        double sumLogRate = 0.0;
        for (int k = 1; k <= 5; k++) {
            double[][] current = UpgmaFixtures.randomSymmetric(LEAF_COUNT, random);
            sumLogRate += Math.log(referenceRate(substitutionDistances, tipDates, current));

            double before = k == 1 ? Double.NaN : perturbation.getRate();
            perturbation.perturb(current, proposed);
            if (k > 1) {
                assertThat(perturbation.getRate()).as("a proposal must not move the rate").isEqualTo(before);
            }

            perturbation.adaptRate();
            assertThat(perturbation.getRate()).as("after %d observations", k)
                    .isCloseTo(Math.exp(sumLogRate / k), within(1e-12));
        }

        // a second adaptRate without a new proposal has nothing to fold in
        double settled = perturbation.getRate();
        perturbation.adaptRate();
        assertThat(perturbation.getRate()).isEqualTo(settled);
    }

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L, 3L})
    void unaffectedDistancesAreCopiedThrough(long seed) {
        Random random = new Random(seed);
        Tree tree = UpgmaFixtures.randomTree(LEAF_COUNT, random);

        UpgmaTaxonSelection selection = new UpgmaTaxonSelection(UpgmaTaxonSelection.Mode.CLADE, tree, 3);
        UpgmaDistancePerturbation perturbation = new UpgmaDistancePerturbation(
                UpgmaDistancePerturbation.Mode.AR1, selection, new SplittableRandom(seed));
        perturbation.setSurrogate(
                UpgmaFixtures.randomSymmetric(LEAF_COUNT, random),
                UpgmaFixtures.randomSymmetric(LEAF_COUNT, random),
                new double[LEAF_COUNT]);

        double[][] current = UpgmaFixtures.randomSymmetric(LEAF_COUNT, random);
        double[][] proposed = new double[LEAF_COUNT][LEAF_COUNT];
        perturbation.perturb(current, proposed);

        int changed = 0;
        for (int i = 0; i < LEAF_COUNT; i++) {
            for (int j = i + 1; j < LEAF_COUNT; j++) {
                boolean affected = selection.isAffected(i) && selection.isAffected(j);
                if (affected) {
                    changed++;
                } else {
                    assertThat(proposed[i][j]).isEqualTo(current[i][j]);
                }
            }
        }
        assertThat(changed).isPositive();
    }

    /** Summed substitution distance over summed path length, over every pair. */
    private static double referenceRate(double[][] substitutionDistances, double[] tipDates, double[][] distances) {
        double substitutions = 0.0;
        double time = 0.0;
        for (int i = 0; i < LEAF_COUNT; i++) {
            for (int j = i + 1; j < LEAF_COUNT; j++) {
                substitutions += substitutionDistances[i][j];
                time += distances[i][j] - tipDates[i] - tipDates[j];
            }
        }
        return substitutions / time;
    }

    /** Log density of {@code to} under the AR(1) step from {@code from}: N(mu + s (from - mu), beta^2 sigma^2). */
    private static double logTransition(double to, double from, double delta, double v, double tipDates,
                                        double rate, double beta) {
        double mean = delta / rate + tipDates;
        double center = mean + Math.sqrt(1.0 - beta * beta) * (from - mean);
        double stepVariance = beta * beta * v / (rate * rate);
        double z = to - center;
        return -0.5 * Math.log(2.0 * Math.PI * stepVariance) - z * z / (2.0 * stepVariance);
    }
}
