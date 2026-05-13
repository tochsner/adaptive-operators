package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import transforms.IntVectorTransform;
import transforms.RealScalarTransform;
import transforms.RealVectorTransform;
import transforms.Transform;

import java.util.ArrayList;
import java.util.List;

public class BasicAdapter extends BEASTObject implements Adapter {

    public final Input<List<Transform<?, ?>>> transformsInput = new Input<>(
            "transform", "", new ArrayList<>()
    );
    public final Input<Boolean> immutableInput = new Input<>(
            "immutable", "", false
    );

    private List<Transform<?, ?>> transforms;
    private boolean immutable;

    @Override
    public void initAndValidate() {
        this.transforms = this.transformsInput.get();
        this.immutable = this.immutableInput.get();
    }

    @Override
    public int getNumImmutable() {
        return this.immutable ? this.transforms.stream().mapToInt(Transform::getDimension).sum() : 0;
    }

    @Override
    public int getNumMutable() {
        return this.immutable ? 0 : this.transforms.stream().mapToInt(Transform::getDimension).sum();
    }

    @Override
    public double[] getImmutable(int nodeId) {
        if (!this.immutable) return new double[0];

        double[] mutable = new double[this.getNumImmutable()];

        int offset = 0;
        for (Transform<?, ?> transform : this.transforms) {
            offset = appendMutable(transform, mutable, offset);
        }

        return mutable;
    }

    @Override
    public double[] getMutable(int nodeId) {
        if (this.immutable) return new double[0];

        double[] mutable = new double[this.getNumMutable()];

        int offset = 0;
        for (Transform<?, ?> transform : this.transforms) {
            offset = appendMutable(transform, mutable, offset);
        }

        return mutable;
    }

    @Override
    public void update(double[] mutable, int nodeId) {
        if (this.immutable) return;

        if (mutable.length != this.getNumMutable()) {
            throw new IllegalArgumentException("Expected " + this.getNumMutable()
                    + " mutable values, but got " + mutable.length);
        }

        int offset = 0;
        for (Transform<?, ?> transform : this.transforms) {
            offset = updateTransform(transform, mutable, offset);
        }
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        if (this.immutable) return 0.0;

        double logCorrection = 0.0;

        for (Transform<?, ?> transform : this.transforms) {
            logCorrection += transform.getLogJacobianCorrection();
        }

        return logCorrection;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>();

        for (Transform<?, ?> transform : this.transforms) {
            stateNodes.add(transform.getStateNode());
        }

        return stateNodes;
    }

    private static int appendMutable(Transform<?, ?> transform, double[] mutable, int offset) {
        if (transform instanceof RealScalarTransform<?> scalarTransform) {
            mutable[offset] = scalarTransform.get();
            return offset + scalarTransform.getDimension();
        }

        if (transform instanceof RealVectorTransform<?> vectorTransform) {
            Double[] values = vectorTransform.get();

            for (Double value : values) {
                mutable[offset++] = value;
            }

            return offset;
        }

        if (transform instanceof IntVectorTransform<?> vectorTransform) {
            Integer[] values = vectorTransform.get();

            for (Integer value : values) {
                mutable[offset++] = value;
            }

            return offset;
        }

        throw new IllegalArgumentException("Unsupported transform type: " + transform.getClass().getName());
    }

    private static int updateTransform(Transform<?, ?> transform, double[] mutable, int offset) {
        if (transform instanceof RealScalarTransform<?> scalarTransform) {
            scalarTransform.set(mutable[offset]);
            return offset + scalarTransform.getDimension();
        }

        if (transform instanceof RealVectorTransform<?> vectorTransform) {
            Double[] values = new Double[vectorTransform.getDimension()];

            for (int i = 0; i < values.length; i++) {
                values[i] = mutable[offset++];
            }

            vectorTransform.set(values);
            return offset;
        }

        throw new IllegalArgumentException("Unsupported transform type: " + transform.getClass().getName());
    }

}
