package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import transforms.RealVectorTransform;

import java.util.List;

public class NodeValueAdapter extends BEASTObject implements Adapter {

    public final Input<RealVectorTransform<?>> nodeValuesInput = new Input<>("nodeValues", "");

    private RealVectorTransform<?> nodeValues;

    @Override
    public void initAndValidate() {
        this.nodeValues = this.nodeValuesInput.get();
    }

    @Override
    public int getNumImmutable() {
        return 0;
    }

    @Override
    public int getNumMutable() {
        return 1;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[] {this.nodeValues.get()[nodeId]};
    }

    @Override
    public void update(double[] mutable, int nodeId) {
        Double[] newValues = this.nodeValues.get().clone();
        newValues[nodeId] = mutable[0];
        this.nodeValues.set(newValues);
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return this.nodeValues.getLogJacobianCorrection(nodeId);
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.nodeValues.getStateNode());
    }

}
