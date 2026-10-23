package adapters;

/**
 * Adapter whose values are measured relative to a reference state, which can be moved to the
 * current state.
 */
public interface ReferenceAdapter extends Adapter {

    /**
     * Makes the current state the reference that later values are measured against.
     */
    void resetReference();

}
