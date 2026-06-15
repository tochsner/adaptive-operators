package transforms;

import beast.base.inference.StateNode;

public non-sealed interface RealVectorTransform<T extends StateNode> extends Transform<T, Double[]> {

    default int getDimension() {
        return this.get().length;
    }

    double getLogJacobianCorrection(int index);

}
