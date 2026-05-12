package adaptiveoperators;

import adapters.Adapter;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.inference.operator.kernel.Transform;

import java.util.ArrayList;
import java.util.List;

public class AdaptiveOperator extends Operator {

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());

    private List<Adapter> adapters;
    private ConditionalSampler sampler;

    private final int numTraining = 10_000;
    private int count = 0;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();

        int totalNumMutable = 0;
        int totalNumImmutable = 0;

        for (Adapter adapter : this.adapters) {
            totalNumMutable += adapter.getNumMutable();
            totalNumImmutable += adapter.getNumImmutable();
        }

        this.sampler = new MultivariateNormalSampler(totalNumImmutable, totalNumMutable);
    }

    @Override
    public double proposal() {
        this.count++;

        double[] oldMutable = this.getMutable();
        double[] oldImmutable = this.getImmutable();

        // record the state in the sampler

        this.sampler.record(oldImmutable, oldMutable);

        if (this.count < this.numTraining) {
            // we are in training phase
            // we don't change the state
            return 0;
        }

        // sample from the conditional distribution

        double[] proposal = this.sampler.sampleConditionally(this.getImmutable());

        // update the adapters

        double logDensityOld = 0.0;
        double logDensityNew = 0.0;

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;

            logDensityOld += adapter.getLogJacobianCorrection();

            double[] proposedMutable = new double[adapter.getNumMutable()];
            System.arraycopy(proposal, idx, proposedMutable, 0, adapter.getNumMutable());
            adapter.update(proposedMutable);

            logDensityNew += adapter.getLogJacobianCorrection();

            idx += adapter.getNumMutable();
        }

        // compute and return the log hastings ratio

        logDensityOld += this.sampler.logDensity(oldImmutable, oldMutable);
        logDensityNew += this.sampler.logDensity(oldImmutable, proposal);

        return logDensityOld - logDensityNew;
    }

    private double[] getMutable() {
        double[] mutable = new double[this.sampler.numValues];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable();
            System.arraycopy(adapterMutable, 0, mutable, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();
        }

        return mutable;
    }

    private double[] getImmutable() {
        double[] immutable = new double[this.sampler.numConditions];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterImmutable = adapter.getImmutable();
            System.arraycopy(adapterImmutable, 0, immutable, idx, adapter.getNumImmutable());
            idx += adapter.getNumImmutable();
        }

        return immutable;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> nodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            nodes.addAll(adapter.listStateNodes());
        }

        return nodes;
    }
}
