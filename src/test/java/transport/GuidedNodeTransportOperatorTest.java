package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.BitSet;
import java.util.Set;
import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThat;

/** Checks which attachments the operator proposes, with and without the current branch. */
class GuidedNodeTransportOperatorTest {

    @Test
    void neverProposesTheCurrentBranchByDefault() {
        ProposalStatistics statistics = runProposals(false, 300);

        assertThat(statistics.proposed()).isPositive();
        assertThat(statistics.keptTopology()).isZero();
    }

    @Test
    void proposesTheCurrentBranchWhenAsked() {
        ProposalStatistics statistics = runProposals(true, 300);

        assertThat(statistics.keptTopology()).isPositive();
        assertThat(statistics.keptTopology()).isLessThan(statistics.proposed());
    }

    private record ProposalStatistics(int proposed, int keptTopology) {}

    /** Proposes repeatedly from the same tree, rejecting every proposal, and counts the topologies. */
    private static ProposalStatistics runProposals(boolean sampleCurrentBranch, int proposals) {
        Alignment data = SmallModel.createAlignment();
        SmallModel model = SmallModel.create(data);

        GuidedNodeTransportOperator operator = new GuidedNodeTransportOperator();
        operator.initByName("tree", model.tree(), "alignment", data, "weightingScheme", "uniform",
                "sampleCurrentBranch", sampleCurrentBranch, "weight", 1.0);

        Supplier<Double> computeCurrentLogLikelihood = () -> {
            model.state().checkCalculationNodesDirtiness();
            return model.posterior().calculateLogP();
        };

        Randomizer.setSeed(7);
        int proposed = 0;
        int keptTopology = 0;

        for (int i = 0; i < proposals; i++) {
            Set<BitSet> before = model.getClades();
            model.state().store(i);

            if (operator.proposal(computeCurrentLogLikelihood, model.state()) != Double.NEGATIVE_INFINITY) {
                proposed++;
                if (model.getClades().equals(before)) {
                    keptTopology++;
                }
            }

            model.recoverAfterRejection(operator);
        }

        return new ProposalStatistics(proposed, keptTopology);
    }
}
