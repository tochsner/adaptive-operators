package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import transforms.RealScalarTransform;
import transforms.RealVectorTransform;
import transforms.Transform;

import java.util.ArrayList;
import java.util.List;

public class BasicAdapter extends BEASTObject implements Adapter {

    public final Input<List<Transform<?, ?>>> transformsInput = new Input<>("transform", "", new ArrayList<>());

    private List<Transform<?, ?>> transforms;

    @Override
    public void initAndValidate() {
        this.transforms = this.transformsInput.get();
    }

    @Override
    public int getNumImmutable() {
        return 0;
    }

    @Override
    public int getNumMutable() {
        return this.transforms.stream().mapToInt(Transform::getDimension).sum();
    }

    @Override
    public double[] getImmutable() {
        return new double[0];
    }

    @Override
    public double[] getMutable() {
        double[] mutable = new double[this.getNumMutable()];

        int offset = 0;
        for (Transform<?, ?> transform : this.transforms) {
            offset = appendMutable(transform, mutable, offset);
        }

        return mutable;
    }

    @Override
    public void update(double[] mutable) {
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
    public double getLogJacobianCorrection() {
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
