package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;

import java.util.LinkedList;
import java.util.Objects;

public class ApproximateFelsenstein {

    int k;
    LinkedList<Integer> cube;
    Alignment alignment;
    RealScalarParam<?> clockRate;

    public ApproximateFelsenstein(int k, LinkedList<Integer> cube, Alignment alignment, RealScalarParam<?> clockRate) {
        this.k = k;
        this.cube = cube;
        this.alignment = alignment;
        this.clockRate = Objects.requireNonNullElse(clockRate, new RealScalarParam<>(1.0, PositiveReal.INSTANCE));
    }

    public double getApproximateFelsenstein(double distances) {

    }

}
