package delayedacceptance;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Input;

import java.util.ArrayList;
import java.util.List;

public abstract class PosteriorApproximation extends BEASTObject {

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());

    protected List<Adapter> adapters;
    protected int numValues;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();

        this.numValues = 0;
        for (Adapter adapter : adapters) {
            this.numValues += adapter.getNumMutable() + adapter.getNumImmutable();
        }
    }

    public abstract double approximateLogPosteriorDifference(double[] previousValues);

    public abstract void registerLogPosteriorDifference(double logPosteriorDifference, double[] previousValues);

    public double[] getCurrentValues() {
        double[] values = new double[this.numValues];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable(0);
            System.arraycopy(adapterMutable, 0, values, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();

            double[] adapterImmutable = adapter.getImmutable(0);
            System.arraycopy(adapterImmutable, 0, values, idx, adapter.getNumImmutable());
            idx += adapter.getNumImmutable();
        }

        return values;
    }

    public abstract boolean isReady();

}
