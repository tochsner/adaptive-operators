package surrogate;

import java.util.List;

public interface SampleGenerator<T> {

    List<T> generateSamples(int numSamples);

}
