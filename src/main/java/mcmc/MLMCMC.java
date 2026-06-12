package mcmc;

import adapters.Adapter;
import beast.base.core.Input;
import beast.base.inference.Operator;
import ml.MLP;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MLMCMC extends MalaMCMC {

    private static final Path RESULTS_PATH = Path.of("results.csv");
    public Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());

    private List<Adapter> adapters;
    private int inputDim = 0;
    private int outputDim = 1;

    private int BURN_IN = 50_000;

    private MLP mlp;
    private double offset = 0.0;


    @Override
    public void initAndValidate() {
        super.initAndValidate();
        removeResults();
        this.adapters = this.adaptersInput.get();

        for (Adapter adapter : this.adapters) {
            this.inputDim += adapter.getNumMutable();
            this.inputDim += adapter.getNumImmutable();
        }

        this.mlp = new MLP(this.inputDim, this.outputDim);
    }

    private void removeResults() {
        try {
            Files.deleteIfExists(RESULTS_PATH);
        } catch (IOException e) {
            throw new IllegalStateException("failed to remove results.csv", e);
        }
    }

    protected Operator propagateState(final long sampleNr) {
        if (sampleNr < this.BURN_IN) {
            return super.propagateState(sampleNr);
        } else if (this.offset == 0.0) {
            this.offset = this.posterior.calculateLogP();
            System.out.println("Offset is " + this.offset);
        }

        double[] state = this.getState();
        this.mlp.record(state, new double[] {oldLogLikelihood - this.offset});

        // if (Randomizer.nextDouble() < 0.001) {
            appendResult(this.mlp.runInference(state)[0] + this.offset, oldLogLikelihood);
        // }

        return super.propagateState(sampleNr);
    }

    private void appendResult(double prediction, double logLikelihood) {
        String line = String.format(Locale.ROOT, "%f,%f%n", prediction, logLikelihood);
        try {
            Files.writeString(
                    RESULTS_PATH,
                    line,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new IllegalStateException("failed to append MLP result", e);
        }
    }

    private double[] getState() {
        double[] state = new double[this.inputDim];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable(0);
            System.arraycopy(adapterMutable, 0, state, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();

            double[] adapterImmutable = adapter.getImmutable(0);
            System.arraycopy(adapterImmutable, 0, state, idx, adapter.getNumImmutable());
        }

        return state;
    }

}
