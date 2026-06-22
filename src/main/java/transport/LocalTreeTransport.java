package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.spec.inference.parameter.RealScalarParam;

import java.util.LinkedList;

public class LocalTreeTransport {

    int k;
    LinkedList<Integer> cube;
    Alignment alignment;
    RealScalarParam<?> clockRate;

    public LocalTreeTransport(int k, LinkedList<Integer> cube, Alignment alignment, RealScalarParam<?> clockRate) {
        this.k = k;
        this.cube = cube;
        this.alignment = alignment;
        this.clockRate = clockRate;
    }

    public double[] transport(double[] currentState) {
        // transport the k distances in currentState to the multivariate Gaussian space
    }

    public double[] transportBack(double[] transportedState) {
        // transport transportedState in the multivariate Gaussian space to k distances
    }

    public double getTransportCorrection(
            double[] currentState, double[] newState, double[] currentTransportedState, double[] newTransportedState
    ) {
        // compute \log l_F(d) - \log l_F(d^\ast) - \log \varphi(z) + \log \varphi(z^\ast)
    }

}
