package adapters;

import beast.base.inference.StateNode;

import java.util.List;

public interface Adapter {

    int getNumImmutable();
    int getNumMutable();

    double[] getImmutable(int nodeId);
    double[] getMutable(int nodeId);

    double update(double[] mutable, int nodeId);

    double getLogJacobianCorrection(int nodeId);

    List<StateNode> listStateNodes();

    /**
     * Stable identifiers for this adapter's mutable coordinates, one per coordinate.
     *
     * <p>Operators that learn a per-coordinate scale key their estimates on these rather than on
     * the coordinate's position. The default is the position itself, which is right whenever
     * position means the same thing from one proposal to the next. Adapters that reshuffle their
     * coordinates in {@link #refresh()}, or whose coordinates depend on the node being worked on,
     * should override this to return something that tracks the underlying quantity instead.
     */
    default int[] getMutableKeys(int nodeId) {
        int[] keys = new int[this.getNumMutable()];

        for (int i = 0; i < keys.length; i++) {
            keys[i] = i;
        }

        return keys;
    }

    default void refresh() {}

}
