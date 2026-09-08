package upgma;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.distance.Distance;
import beast.base.evolution.distance.JukesCantorDistance;
import beast.base.evolution.tree.Tree;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.assertj.core.api.Assertions.withinPercentage;

/**
 * The surrogate has to sit where BEAST's Jukes-Cantor distance sits and be as wide as the
 * pairwise likelihood is curved there.
 */
class UpgmaJukesCantorSurrogateTest {

    private static final int LEAF_COUNT = 6;
    private static final int SITES = 400;

    @Test
    void distancesMatchBeastsJukesCantorDistance() {
        Random random = new Random(1L);
        Tree tree = UpgmaFixtures.randomTree(LEAF_COUNT, random);
        Alignment alignment = randomAlignment(tree, random);

        JukesCantorDistance reference = new JukesCantorDistance();
        reference.setPatterns(alignment);

        UpgmaJukesCantorSurrogate surrogate = new UpgmaJukesCantorSurrogate(alignment, tree);
        for (int i = 0; i < LEAF_COUNT; i++) {
            for (int j = 0; j < LEAF_COUNT; j++) {
                int taxonI = alignment.getTaxonIndex(tree.getNode(i).getID());
                int taxonJ = alignment.getTaxonIndex(tree.getNode(j).getID());
                assertThat(surrogate.distances()[i][j])
                        .isCloseTo(reference.pairwiseDistance(taxonI, taxonJ), within(1e-12));
            }
        }
    }

    /** The variance is the inverse negative curvature of the pairwise log-likelihood at its peak. */
    @Test
    void varianceIsTheLaplaceCurvatureOfThePairwiseLikelihood() {
        Random random = new Random(2L);
        Tree tree = UpgmaFixtures.randomTree(LEAF_COUNT, random);
        Alignment alignment = randomAlignment(tree, random);

        UpgmaJukesCantorSurrogate surrogate = new UpgmaJukesCantorSurrogate(alignment, tree);
        for (int i = 0; i < LEAF_COUNT; i++) {
            for (int j = i + 1; j < LEAF_COUNT; j++) {
                double mismatches = mismatches(alignment, tree, i, j);
                double delta = surrogate.distances()[i][j];
                assertThat(mismatches).as("the random alignment should not produce identical rows").isPositive();

                // central second difference of the pairwise log-likelihood in delta
                double h = 1e-4;
                double curvature = (logLikelihood(delta + h, mismatches) - 2.0 * logLikelihood(delta, mismatches)
                        + logLikelihood(delta - h, mismatches)) / (h * h);

                assertThat(surrogate.variances()[i][j])
                        .as("pair %d %d", i, j)
                        .isCloseTo(-1.0 / curvature, withinPercentage(0.01));
            }
        }
    }

    @Test
    void identicalAndSaturatedPairsStayFinite() {
        Tree tree = UpgmaFixtures.parse("(T0:1.0,(T1:0.5,T2:0.5):0.5):0.0;");
        List<Sequence> sequences = List.of(
                new Sequence("T0", "ACGTACGTAC"),
                new Sequence("T1", "ACGTACGTAC"),
                new Sequence("T2", "CATGCATGCA"));
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        UpgmaJukesCantorSurrogate surrogate = new UpgmaJukesCantorSurrogate(alignment, tree);

        assertThat(surrogate.distances()[0][1]).isEqualTo(0.0);
        assertThat(surrogate.variances()[0][1]).isPositive().isFinite();

        assertThat(surrogate.distances()[0][2]).isEqualTo(Distance.Base.MAX_DISTANCE);
        assertThat(surrogate.variances()[0][2]).isGreaterThan(1.0).isFinite();
    }

    /** Jukes-Cantor pairwise log-likelihood {@code k log p + (L - k) log(1 - p)} for nucleotides. */
    private static double logLikelihood(double delta, double mismatches) {
        double p = 0.75 * (1.0 - Math.exp(-delta / 0.75));
        return mismatches * Math.log(p) + (SITES - mismatches) * Math.log(1.0 - p);
    }

    private static double mismatches(Alignment alignment, Tree tree, int i, int j) {
        int taxonI = alignment.getTaxonIndex(tree.getNode(i).getID());
        int taxonJ = alignment.getTaxonIndex(tree.getNode(j).getID());
        double mismatches = 0.0;
        for (int pattern = 0; pattern < alignment.getPatternCount(); pattern++) {
            int[] states = alignment.getPattern(pattern);
            if (states[taxonI] != states[taxonJ]) {
                mismatches += alignment.getPatternWeight(pattern);
            }
        }
        return mismatches;
    }

    /** Unrelated random nucleotide rows per leaf, roughly a third of the sites shared with row 0. */
    static Alignment randomAlignment(Tree tree, Random random) {
        String bases = "ACGT";
        char[] first = new char[SITES];
        for (int site = 0; site < SITES; site++) {
            first[site] = bases.charAt(random.nextInt(4));
        }

        List<Sequence> sequences = new ArrayList<>();
        for (int leaf = 0; leaf < tree.getLeafNodeCount(); leaf++) {
            char[] row = first.clone();
            double divergence = 0.05 + 0.4 * random.nextDouble();
            for (int site = 0; site < SITES; site++) {
                if (random.nextDouble() < divergence) {
                    row[site] = bases.charAt(random.nextInt(4));
                }
            }
            sequences.add(new Sequence(tree.getNode(leaf).getID(), new String(row)));
        }

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");
        return alignment;
    }
}
