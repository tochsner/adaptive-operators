package transport;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;

import java.util.*;

public class PartialCubeTransportOperator extends Operator {

    public Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    public final Input<RealScalarParam<?>> clockRateInput = new Input<>("clockRate", "", null, Input.Validate.OPTIONAL);

    // number of distances to consider
    int WINDOW_SIZE = 2;
    int BURN_IN = 1000;

    double scaleFactor = 1.0;
    Random random = new Random();

    Tree tree;
    Alignment alignment;
    RealScalarParam<?> clockRate;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();
        this.clockRate = this.clockRateInput.get();
    }

    int count = 0;

    @Override
    public double proposal() {
        this.count++;

        if (this.count < BURN_IN) return Double.NEGATIVE_INFINITY;
        else if (this.count == BURN_IN) System.out.println("Start transporting cubes");

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

        LocalTreeTransport localTreeTransport = new LocalTreeTransport(
                taxaIds, this.alignment, this.clockRate, 2*this.tree.getRoot().getHeight()
        );

        // propose move

        double[] currentCubeDistances = currentCubeDistances = TreeUtils.getCubeDistances(this.tree, cube);

        double[] currentState = new double[WINDOW_SIZE];
        System.arraycopy(currentCubeDistances, k, currentState, 0, WINDOW_SIZE);

        double[] currentTransportedState = localTreeTransport.transport(currentState);
        double[] newTransportedState = new double[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            newTransportedState[i] = currentTransportedState[i] + Randomizer.nextGaussian() * this.scaleFactor;
        }

        double[] newState = localTreeTransport.transportBack(newTransportedState);

        if (!this.validatedDistances(newState)) {
            System.out.println("Invalid distances");
            return Double.NEGATIVE_INFINITY;
        }

        // update the tree

        this.updateDistances(cube, k, newState);

        // compute log HR correction

        double logHR = localTreeTransport.getTransportCorrection(currentState, newState, currentTransportedState, newTransportedState);

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
