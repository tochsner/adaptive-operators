package transforms;

import beast.base.inference.StateNode;

public sealed interface Transform<T extends StateNode, V> permits RealScalarTransform, RealVectorTransform {

    int getDimension();
    V get();
    void set(V value);
    double getLogJacobianCorrection();
    T getStateNode();

}
