package adapters;

public interface Adapter {

    int getNumImmutable();
    int getNumMutable();

    double[] getImmutable();
    double[] getMutable();

    void update(double[] mutable);

    double getLogJacobianCorrection();

}
