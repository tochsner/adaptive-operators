package adapters;

import beast.base.inference.StateNode;

import java.util.Collection;
import java.util.List;

public interface Adapter {

    int getNumImmutable();
    int getNumMutable();

    double[] getImmutable();
    double[] getMutable();

    void update(double[] mutable);

    double getLogJacobianCorrection();

    List<StateNode> listStateNodes();

}
