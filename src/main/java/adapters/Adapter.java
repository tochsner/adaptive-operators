package adapters;

import beast.base.inference.StateNode;

import java.util.List;

public interface Adapter {

    int getNumImmutable();
    int getNumMutable();

    double[] getImmutable(int nodeId);
    double[] getMutable(int nodeId);

    void update(double[] mutable, int nodeId);

    double getLogJacobianCorrection(int nodeId);

    List<StateNode> listStateNodes();

}
