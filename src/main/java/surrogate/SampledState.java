package surrogate;

import java.util.List;

public record SampledState(List<double[]> mutableStates, List<double[]> immutableStates, double logPosterior) {
}
