package transforms;

import beast.base.spec.inference.parameter.IntVectorParam;

public non-sealed interface IntVectorTransform<T extends IntVectorParam<?>> extends Transform<T, Integer[]> {

    default int getDimension() {
        return this.get().length;
    }

}