package transport;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.sitemodel.SiteModelInterface;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
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
    int BURN_IN = 900;

    double scaleFactor = 1.0;
    Random random = new Random(0);

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
        this.count++;
        if (this.count < BURN_IN) return Double.NEGATIVE_INFINITY;
        else if (this.count == BURN_IN) System.out.println("Start transport");

        // choose new compatible cube

        LinkedList<Integer> cube = TreeUtils.getRandomCompatibleCube(this.tree.getRoot(), this.random);

        // choose window

        int k = Randomizer.nextInt(cube.size() - this.WINDOW_SIZE);

        // obtain taxa

        List<String> taxaIds = new ArrayList<>();
        List<Double> taxaHeights = new ArrayList<>();
        for (int i = k; i <= k + WINDOW_SIZE; i++) {
            int nodeNr = cube.get(i);
            Node node = this.tree.getNode(nodeNr);
            String taxonId = this.tree.getTaxonId(node);
            taxaIds.add(taxonId);
            taxaHeights.add(node.getHeight());
        }

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

        // set up transport

        double maxHeight = 4* Arrays.stream(currentState).max().orElseThrow();
        Local3TaxaTransport localTreeTransport = new Local3TaxaTransport(
                taxaIds, taxaHeights, this.alignment, this.siteModel, this.clockRate, maxHeight
        );
        int extensionSize = (currentState.length - 2) / 2;
        double[] currentCentralState = this.getCentralState(currentState, extensionSize);

        // print values

        boolean debug = false;

        if (debug) {
            String trees = "";

            System.out.println(465 + " " + tree.getNode(465).getID());
            System.out.println(432 + " " + tree.getNode(432).getID());
            System.out.println(677 + " " + tree.getNode(677).getID());
            System.out.println(827 + " " + tree.getNode(827).getID());
            System.out.println(682 + " " + tree.getNode(682).getID());
            System.out.println(961 + " " + tree.getNode(961).getID());

            for (int i = 0; i < 100; i++) {
                double[] shiftedCentralState = new double[2];
                shiftedCentralState[0] = (i + 1.0) * maxHeight / 100;

                for (int j = 0; j < 100; j++) {
                    shiftedCentralState[1] = (j + 1.0) * maxHeight / 100;
                    double[] centralState = localTreeTransport.unshiftDistances(shiftedCentralState);
                    double[] fullState = this.withCentralState(currentState, centralState, extensionSize);

                    this.updateDistances(cube, k, fullState);

                    double real = computeCurrentLogLikelihood.get();
                    double approx = localTreeTransport.felsensteinLogPDF(centralState);

                    System.out.println("grid," + centralState[0] + "," + centralState[1] + "," + real + "," + approx);

                    if (i == 5 && j == 35 || i == 5 && j == 57) {
                        trees += centralState[0] + "," + centralState[1] + ",full," + this.tree.toString() + "\n";
                        trees += centralState[0] + "," + centralState[1] + ",partial," + localTreeTransport.getTree(centralState).toString() + "\n";
                    }
                }
            }


            System.arraycopy(currentCubeDistances, k, currentState, 0, WINDOW_SIZE);
            currentCentralState = this.getCentralState(currentState, extensionSize);
            System.out.println("current," + currentCentralState[0] + "," + currentCentralState[1] + "," + currentCentralState[0] + "," + currentCentralState[1]);

            for (int m = 0; m < 1000; m++) {
                double[] currentTransportedState = localTreeTransport.transport(currentCentralState);

                double[] newTransportedState = new double[currentTransportedState.length];
                for (int i = 0; i < newTransportedState.length; i++) {
                    newTransportedState[i] = currentTransportedState[i] + 2.0 * Randomizer.nextGaussian();
                }

                double[] newState = localTreeTransport.transportBack(newTransportedState);
                System.out.println("new," + newState[0] + "," + newState[1] + "," + newTransportedState[0] + "," + newTransportedState[1]);
            }

            for (int i = k; i <= k + WINDOW_SIZE; i++) {
                int nodeNr = cube.get(i);
                Node node = this.tree.getNode(nodeNr);
                System.out.println(node.getHeight());
            }

            System.out.println(trees);

            System.exit(0);
        }

        double[] currentTransportedState = null;
        try {
            currentTransportedState = localTreeTransport.transport(currentCentralState);
        } catch (RuntimeException e) {
            System.out.println("B" + e);
            return Double.NEGATIVE_INFINITY;
        }

        double[] newTransportedState = new double[currentTransportedState.length];
        for (int i = 0; i < newTransportedState.length; i++) {
            newTransportedState[i] = currentTransportedState[i] + Randomizer.nextGaussian() * this.scaleFactor;
        }

        double[] newCentralState = null;
        try {
            newCentralState = localTreeTransport.transportBack(newTransportedState);
        } catch (NoBracketingException e) {
            System.out.println("C" + e);
            return Double.NEGATIVE_INFINITY;
        }
        double[] newState = this.withCentralState(currentState, newCentralState, extensionSize);

        if (!this.validatedDistances(newState)) {
            System.out.println("Invalid distances");
            return Double.NEGATIVE_INFINITY;
        }

        // update the tree

        this.updateDistances(cube, k, newState);

        // compute log HR correction

        double logHR = localTreeTransport.getTransportCorrection(currentCentralState, newCentralState, currentTransportedState, newTransportedState);

        return logHR;
    }

    private double[] getCentralState(double[] distances, int extensionSize) {
        return new double[] {distances[extensionSize], distances[extensionSize + 1]};
    }

    private double[] withCentralState(double[] distances, double[] centralState, int extensionSize) {
        double[] fullState = distances.clone();
        fullState[extensionSize] = centralState[0];
        fullState[extensionSize + 1] = centralState[1];
        return fullState;
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
        if (this.count < 2*BURN_IN) return;

        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.scaleFactor);
        this.scaleFactor = Math.exp(delta);
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}
