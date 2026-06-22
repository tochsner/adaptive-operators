package transport;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.sitemodel.SiteModelInterface;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.inference.StateNode;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.exception.NoBracketingException;
import slice.SliceOperator;

import java.util.*;
import java.util.function.Supplier;

public class PartialCubeTransportOperator extends SliceOperator {

    public Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    final public Input<SiteModelInterface> siteModelInput = new Input<>("siteModel", "site model for leafs in the beast.tree");
    public final Input<RealScalarParam<?>> clockRateInput = new Input<>("clockRate", "", null, Input.Validate.OPTIONAL);

    // number of distances to consider
    int WINDOW_SIZE = 2;

    double scaleFactor = 1.0;
    Random random = new Random();

    Tree tree;
    Alignment alignment;
    SiteModelInterface siteModel;
    RealScalarParam<?> clockRate;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();
        this.siteModel = this.siteModelInput.get();
        this.clockRate = this.clockRateInput.get();
    }

    int count = 0;

    @Override
    public double proposal() {
        return 0;
    }

    @Override
    public double proposal(Supplier<Double> computeCurrentLogLikelihood, State state) {
        if (count++ < 100010) return Double.NEGATIVE_INFINITY;

        // choose new compatible cube

        LinkedList<Integer> cube = TreeUtils.getRandomCompatibleCube(this.tree.getRoot(), this.random);

        // choose window

        int k = Randomizer.nextInt(cube.size() - this.WINDOW_SIZE);

        // obtain taxa names

        List<String> taxaIds = new ArrayList<>();
        for (int i = k; i <= k + WINDOW_SIZE; i++) {
            int nodeNr = cube.get(i);
            Node node = this.tree.getNode(nodeNr);
            String taxonId = this.tree.getTaxonId(node);
            taxaIds.add(taxonId);
        }

        // set up transport

        double maxHeight = 4*this.tree.getRoot().getHeight();
        LocalTreeTransport localTreeTransport = new LocalTreeTransport(
                taxaIds, this.alignment, this.siteModel, this.clockRate, maxHeight
        );

        // propose move

        double[] currentCubeDistances = null;
        try {
            currentCubeDistances = TreeUtils.getCubeDistances(this.tree, cube);
        } catch (Exception e) {
            System.out.println("A" + e);
            throw new RuntimeException(e);
        }

        double[] currentState = new double[WINDOW_SIZE];
        System.arraycopy(currentCubeDistances, k, currentState, 0, WINDOW_SIZE);

        // print values

        for (int i = 0; i < 100; i++) {
            currentState[0] = (i+1.0) * maxHeight / 100;

            for (int j = 0; j < 100; j++) {
                currentState[1] = (j+1.0) * maxHeight / 100;

                this.updateDistances(cube, k, currentState.clone());

                double real = computeCurrentLogLikelihood.get();
                double approx = localTreeTransport.felsensteinLogPDF(currentState.clone());

                System.out.println(currentState[0] + "," + currentState[1] + "," + real + "," + approx);
            }
        }

        System.exit(0);

        double[] currentTransportedState = null;
        try {
            currentTransportedState = localTreeTransport.transport(currentState);
        } catch (RuntimeException e) {
            System.out.println("B" + e);
            return Double.NEGATIVE_INFINITY;
        }

        double[] newTransportedState = new double[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            newTransportedState[i] = currentTransportedState[i] + Randomizer.nextGaussian() * this.scaleFactor;
        }

        double[] newState = null;
        try {
            newState = localTreeTransport.transportBack(newTransportedState);
        } catch (NoBracketingException e) {
            System.out.println("C" + e);
            return Double.NEGATIVE_INFINITY;
        }

        if (!this.validatedDistances(newState)) {
            System.out.println("Invalid distances");
            return Double.NEGATIVE_INFINITY;
        }

        // update the tree

        this.updateDistances(cube, k, newState);

        // compute log HR correction

        double logHR = localTreeTransport.getTransportCorrection(currentState, newState, currentTransportedState, newTransportedState);

        System.out.println("OK " + logHR);

        return logHR;
    }

    private boolean validatedDistances(double[] newState) {
        for (Double value : newState) {
            if (value == 0.0 || !Double.isFinite(value)) return false;
        }

        if (Arrays.stream(newState).distinct().count() != newState.length) return false;

        return true;
    }

    private void updateDistances(LinkedList<Integer> cube, int k, double[] distances) {
        for (int i = k; i < k + WINDOW_SIZE; i++) {
            Node nodeA = tree.getNode(cube.get(i));
            Node nodeB = tree.getNode(cube.get(i + 1));
            TreeUtils.deterministicallyChangeNodeDistance(nodeA, nodeB, distances[i - k], cube);
        }
    }

    @Override
    public double getCoercableParameterValue() {
        return this.scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("scaleFactor must be finite and positive");
        }

        this.scaleFactor = value;
    }

    @Override
    public void optimize(double logAlpha) {
        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.scaleFactor);
        this.scaleFactor = Math.exp(delta);
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
