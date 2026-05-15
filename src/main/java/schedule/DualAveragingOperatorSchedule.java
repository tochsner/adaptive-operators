package schedule;

import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DualAveragingOperatorSchedule extends OperatorSchedule {

    double GAMMA = 0.05;
    int T_0 = 10;
    double KAPPA = 0.75;

    Map<Operator, Double> MUs = new HashMap<>();
    Map<Operator, List<Double>> acceptanceProbabilities = new HashMap<>();
    Map<Operator, Double> Hs = new HashMap<>();

    @Override
    public double calcDelta(Operator operator, double logAlpha) {
        if (autoOptimizeDelayCount < autoOptimizeDelay || !isAutoOptimise()) {
            autoOptimizeDelayCount++;
            return 0;
        }

        double mu = this.MUs.computeIfAbsent(operator, x -> Math.log(10 * x.getCoercableParameterValue()));
        double delta = operator.getTargetAcceptanceProbability();
        List<Double> operatorAcceptanceProbabilities = this.acceptanceProbabilities.computeIfAbsent(
                operator, x -> new ArrayList<>()
        );

        operatorAcceptanceProbabilities.add(Math.exp(Math.min(logAlpha, 0)));
        int t = operatorAcceptanceProbabilities.size();

        // compute y_{t+1}

        double H = Hs.computeIfAbsent(operator, x -> 0.0);
        H += (delta - operatorAcceptanceProbabilities.get(t - 1));
        Hs.put(operator, H);

        double y = mu - (Math.sqrt(t) / GAMMA) * (1.0 / (t + T_0)) * H;

        // compute new log value

        double logX = Math.log(operator.getCoercableParameterValue());
        double eta_t = Math.pow(t, -this.KAPPA);
        double newLogX = eta_t * y + (1.0 - eta_t) * logX;

        // return delta

        return newLogX - logX;
    }

}
