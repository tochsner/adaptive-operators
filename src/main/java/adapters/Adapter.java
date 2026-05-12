package adapters;

public interface Adapter {

    double[] getImmutable();
    double[] getMutable();

    void update(double[] mutable);

}
