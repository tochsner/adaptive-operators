package transforms;

import beast.base.inference.StateNode;
import beast.base.spec.inference.parameter.RealVectorParam;
import beast.base.spec.type.RealVector;

public non-sealed interface RealVectorTransform<T extends RealVectorParam<?>> extends Transform<T, Double[]> {

    default int getDimension() {
        return this.get().length;
    }

}