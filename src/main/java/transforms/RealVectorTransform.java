package transforms;

import beast.base.spec.inference.parameter.RealVectorParam;

public non-sealed interface RealVectorTransform<T extends RealVectorParam<?>> extends Transform<T, Double[]> {

    default int getDimension() {
        return this.get().length;
    }

    double getLogJacobianCorrection(int index);

}