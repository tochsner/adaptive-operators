package weightoptimization;

import beast.base.core.BEASTObject;
import beast.base.core.BEASTObjectStore;
import beast.base.core.Input;
import beast.base.inference.Operator;

import java.util.*;

public class RunningAverageScheme extends BEASTObject implements WeightScheme {

    public final Input<Integer> lagInput = new Input<>(
            "lag", "", 2
    );
    public final Input<Double> temperatureInput = new Input<>(
            "temperature", "", 0.001
    );
    public final Input<Double> mixingFactorInput = new Input<>(
            "mixingFactor", "", 0.001
    );

    List<Double> lastLogAlphas;
    List<Operator> lastPickedOperators;

    double sum = 0.0;
    int count = 0;

    @Override
    public void initAndValidate() {
        this.lastLogAlphas = new LinkedList<>();
        this.lastPickedOperators = new LinkedList<>();
    }

    @Override
    public void optimizeWeights(List<Operator> operators, double logAlpha, int lastOperatorIdx) {
        if (!Double.isFinite(logAlpha)) return;

        this.count++;
        this.sum += logAlpha;

        if (this.count < 50_000) return;

        int lag = this.lagInput.get();
        double mixingFactor = this.mixingFactorInput.get();

        // update last log alpha stack

        this.lastLogAlphas.add(logAlpha);
        if (this.lastLogAlphas.size() == lag + 1) {
            this.lastLogAlphas.removeFirst();
        }

        // update last used operators

        this.lastPickedOperators.add(operators.get(lastOperatorIdx));
        if (this.lastPickedOperators.size() == lag + 1) {
            this.lastPickedOperators.removeFirst();
        }

        // update weight of last picked operator

        if (this.lastPickedOperators.size() == lag) {
            Operator operatorToOptimize = this.lastPickedOperators.getFirst();

            double accumulatedLogAlpha = 0;
            double normalization = 0;
            for (int i = 0; i < this.lastLogAlphas.size(); i++) {
                accumulatedLogAlpha += this.lastLogAlphas.get(i) * (i + 1);
                normalization += i + 1;
            }

            double oldWeight = operatorToOptimize.getWeight();
            double newWeight = (1.0 - mixingFactor) * oldWeight + mixingFactor * accumulatedLogAlpha / normalization;

            operatorToOptimize.m_pWeight.setValue(newWeight, BEASTObjectStore.INSTANCE.getBEASTObject(operatorToOptimize.m_pWeight));
        }
    }

    @Override
    public double[] getOperatorProbabilities(List<Operator> operators) {
        int numOperators = operators.size();

        double[] operatorWeights = new double[numOperators];

        // apply softmax

        double mean = this.sum / this.count;
        double weightSum = 0;

        for (int i = 0; i < numOperators; i++) {
            operatorWeights[i] = Math.exp(
                    this.temperatureInput.get() * (operators.get(i).getWeight() - mean)
            );
            weightSum += operatorWeights[i];
        }

        // normalize

        for (int i = 0; i < numOperators; i++) {
            operatorWeights[i] = operatorWeights[i] / weightSum;
        }

        return operatorWeights;
    }

}
