package transforms;

import beast.base.spec.inference.parameter.RealScalarParam;

public non-sealed interface RealScalarTransform<T extends RealScalarParam<?>> extends Transform<T, Double> {

    default int getDimension() {
        return 1;
    }

}
