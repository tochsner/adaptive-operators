package weightoptimization;

import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.*;

public class AdaptiveWeightOperator extends Operator {

    final public Input<List<Operator>> operatorsInput = new Input<>(
            "operator", "list of operators to select from", new ArrayList<>()
    );
    final public Input<WeightScheme> weightSchemeInput = new Input<>(
            "weightScheme", ""
    );
    final public Input<Integer> burnInInput = new Input<>(
            "burnIn", "", 10_000
    );

    List<Operator> operators;
    WeightScheme weightScheme;

    int count;
    private int lastOperatorIdx;

    @Override
    public void initAndValidate() {
        this.operators = this.operatorsInput.get();
        this.weightScheme = this.weightSchemeInput.get();
    }

    @Override
    public double proposal() {
        // get cumulative probabilities

        double[] probabilities = this.weightScheme.getOperatorProbabilities(this.operators);
        double[] cumulativeProbabilities = new double[this.operators.size()];

        double cumProb = 0;
        for (int i = 0; i < this.operators.size(); i++) {
            if (this.count < this.burnInInput.get()) {
                cumProb += 1.0 / this.operators.size();
            } else {
                cumProb += probabilities[i];
            }

            cumulativeProbabilities[i] = cumProb;
        }

        // sample an operator

        this.lastOperatorIdx = Randomizer.binarySearchSampling(cumulativeProbabilities);
        Operator operator = this.operators.get(this.lastOperatorIdx);

        return operator.proposal();
    }

    @Override
    public void optimize(double logAlpha) {
        this.count++;

        if (this.count < this.burnInInput.get()) return;

        this.weightScheme.optimizeWeights(this.operators, logAlpha, lastOperatorIdx);
    }

    @Override
    public List<StateNode> listStateNodes() {
        Set<StateNode> stateNodes = new HashSet<>();

        for (Operator operator : this.operators) {
            stateNodes.addAll(operator.listStateNodes());
        }

        return stateNodes.stream().toList();
    }
}
