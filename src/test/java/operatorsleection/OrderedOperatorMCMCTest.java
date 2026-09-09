package operatorsleection;

import beast.base.inference.Operator;
import operatorsleection.OrderedOperatorMCMC.Segment;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Pins the operator cycle of {@link OrderedOperatorMCMC} to its definition: the cycle is a fixed
 * list of segments built by capped round robin, each operator gets exactly its integer weight per
 * cycle, no segment exceeds the cap, and the position wraps around after {@code C} steps.
 */
class OrderedOperatorMCMCTest {

    private static final int UNCAPPED = Integer.MAX_VALUE;

    @Test
    void roundsWeightsToIntegerWeights() {
        List<Operator> operators = operators(2.0, 1.0, 3.0);

        assertThat(OrderedOperatorMCMC.integerWeights(operators, 0)).containsExactly(2, 1, 3);
    }

    @Test
    void keepsTheRequestedNumberOfDecimalPlaces() {
        List<Operator> operators = operators(0.25, 0.5, 1.0);

        assertThat(OrderedOperatorMCMC.integerWeights(operators, 2)).containsExactly(25, 50, 100);
        assertThat(OrderedOperatorMCMC.integerWeights(operators, 1)).containsExactly(3, 5, 10);
    }

    @Test
    void rejectsWeightsThatRoundToZero() {
        List<Operator> operators = operators(0.25, 1.0);

        assertThatThrownBy(() -> OrderedOperatorMCMC.integerWeights(operators, 0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("weightPrecision");
    }

    @Test
    void usesOneSegmentPerOperatorWithoutACap() {
        assertThat(OrderedOperatorMCMC.cappedRoundRobin(new int[]{2, 1, 3}, UNCAPPED))
                .containsExactly(new Segment(0, 2), new Segment(1, 1), new Segment(2, 3));
    }

    @Test
    void spreadsHeavyOperatorsOverSeveralSweeps() {
        assertThat(OrderedOperatorMCMC.cappedRoundRobin(new int[]{30, 5, 10}, 10))
                .containsExactly(
                        new Segment(0, 10), new Segment(1, 5), new Segment(2, 10),
                        new Segment(0, 10),
                        new Segment(0, 10)
                );
    }

    @Test
    void capOfOneGivesAStepwiseInterleaving() {
        assertThat(OrderedOperatorMCMC.cappedRoundRobin(new int[]{3, 1, 2}, 1))
                .containsExactly(
                        new Segment(0, 1), new Segment(1, 1), new Segment(2, 1),
                        new Segment(0, 1), new Segment(2, 1),
                        new Segment(0, 1)
                );
    }

    @Test
    void preservesTheWeightOfEveryOperatorUnderAnyCap() {
        int[] weights = {17, 1, 4, 23, 9};

        for (int maxDwell = 1; maxDwell <= 25; maxDwell++) {
            int[] served = new int[weights.length];
            for (Segment segment : OrderedOperatorMCMC.cappedRoundRobin(weights, maxDwell)) {
                assertThat(segment.length()).as("segment length at cap " + maxDwell).isBetween(1, maxDwell);
                served[segment.operatorIndex()] += segment.length();
            }
            assertThat(served).as("steps per operator at cap " + maxDwell).containsExactly(weights);
        }
    }

    @Test
    void rejectsACapBelowOne() {
        assertThatThrownBy(() -> OrderedOperatorMCMC.cappedRoundRobin(new int[]{1}, 0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("maxDwell");
    }

    @Test
    void dwellsOnEachOperatorForItsWeightAndThenWrapsAround() {
        List<Operator> operators = operators(2.0, 1.0, 3.0);
        OrderedOperatorMCMC mcmc = new OrderedOperatorMCMC();
        mcmc.setUpCycle(operators, 0, UNCAPPED);

        List<Operator> expectedCycle = List.of(
                operators.get(0), operators.get(0),
                operators.get(1),
                operators.get(2), operators.get(2), operators.get(2)
        );

        assertCycles(mcmc, expectedCycle);
    }

    @Test
    void followsTheSegmentsWhenCapped() {
        List<Operator> operators = operators(3.0, 1.0, 2.0);
        OrderedOperatorMCMC mcmc = new OrderedOperatorMCMC();
        mcmc.setUpCycle(operators, 0, 2);

        // sweep one: A×2 B×1 C×2, sweep two: A×1
        List<Operator> expectedCycle = List.of(
                operators.get(0), operators.get(0),
                operators.get(1),
                operators.get(2), operators.get(2),
                operators.get(0)
        );

        assertCycles(mcmc, expectedCycle);
    }

    @Test
    void usesTheSingleOperatorEveryStep() {
        List<Operator> operators = operators(1.0);
        OrderedOperatorMCMC mcmc = new OrderedOperatorMCMC();
        mcmc.setUpCycle(operators, 0, UNCAPPED);

        for (int step = 0; step < 5; step++) {
            assertThat(mcmc.nextOperator()).isSameAs(operators.get(0));
            assertThat(mcmc.getPosition()).isEqualTo(1);
        }
    }

    private static void assertCycles(OrderedOperatorMCMC mcmc, List<Operator> expectedCycle) {
        for (int cycle = 0; cycle < 3; cycle++) {
            for (int c = 1; c <= expectedCycle.size(); c++) {
                assertThat(mcmc.getPosition()).as("position before step").isEqualTo(c);
                assertThat(mcmc.nextOperator()).as("operator at position " + c).isSameAs(expectedCycle.get(c - 1));
            }
        }
    }

    private static List<Operator> operators(double... weights) {
        List<Operator> operators = new ArrayList<>();
        for (double weight : weights) {
            Operator operator = new StubOperator();
            operator.initByName("weight", weight);
            operators.add(operator);
        }
        return operators;
    }

    /** An operator that never proposes anything; only its weight matters here. */
    private static class StubOperator extends Operator {
        @Override
        public void initAndValidate() {
        }

        @Override
        public double proposal() {
            return 0;
        }
    }
}
