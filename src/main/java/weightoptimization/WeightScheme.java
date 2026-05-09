package weightoptimization;

import beast.base.inference.Operator;

import java.util.List;

public interface WeightScheme {

    void optimizeWeights(List<Operator> operators, double logAlpha, int lastOperatorIdx);
    double[] getOperatorProbabilities(List<Operator> operators);

}
