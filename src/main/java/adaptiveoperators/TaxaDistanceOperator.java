package adaptiveoperators;

import adapters.TreeUtils;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TaxaDistanceOperator extends TreeOperator {

    public final Input<Integer> numberOfPairsInput = new Input<>(
            "numberOfPairs",
            "number of unordered taxon pairs to learn distance proposals for",
            100);

    public final Input<List<Operator>> alternativeOperatorsInput = new Input<>(
            "alternativeOperator",
            "",
            new ArrayList<>());

    public final Input<Boolean> neuralInput = new Input<>(
            "neural",
            "whether to model the distances jointly with a neural network instead of "
                    + "independent log-normals",
            true);

    private static final int BURN_IN = 20_000;
    private static final int START_TRAINING = 100_000;
    private static final int END_TRAINING = 900_000;

    private Tree tree;
    private List<Operator> alternativeOperators;

    private TaxonPair[] taxonPairs;
    private TaxaDistanceModel model;
    private Random random;
    private int count = 0;

    double scaleFactor = 1.0;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alternativeOperators = this.alternativeOperatorsInput.get();

        int leafCount = this.tree.getLeafNodeCount();
        int numberOfPairs = this.numberOfPairsInput.get();

        if (leafCount < 2) {
            throw new IllegalArgumentException("TaxaDistanceOperator requires at least two taxa");
        }

        if (numberOfPairs < 1) {
            throw new IllegalArgumentException("numberOfPairs must be at least one");
        }

        this.taxonPairs = sampleTaxonPairs(leafCount, numberOfPairs);

        double[] offsets = new double[this.taxonPairs.length];
        for (int i = 0; i < offsets.length; i++) {
            TaxonPair pair = this.taxonPairs[i];
            offsets[i] = Math.abs(
                    this.tree.getNode(pair.firstTaxon).getHeight()
                            - this.tree.getNode(pair.secondTaxon).getHeight());
        }

        this.model = this.neuralInput.get()
                ? new NeuralLogNormalModel(offsets)
                : new LogNormalModel(offsets);
        this.random = new Random(Randomizer.nextLong());
    }

    @Override
    public double proposal() {
        this.count++;

        if (this.count < BURN_IN) {
            return proposeAlternativeOperator();
        }

        double[] distances = currentDistances();
        int pairIndex = Randomizer.nextInt(this.taxonPairs.length);

        if (this.count < END_TRAINING) {
            this.model.record(distances, pairIndex);
        }

        if (this.count < START_TRAINING) {
            return proposeAlternativeOperator();
        } else if (this.count == START_TRAINING) {
            System.out.println("Adaptive tree starts");
        } else if (this.count == END_TRAINING) {
            System.out.println("Adaptive tree ends");
        }

        TaxonPair pair = this.taxonPairs[pairIndex];
        Node nodeA = this.tree.getNode(pair.firstTaxon);
        Node nodeB = this.tree.getNode(pair.secondTaxon);
        double oldDistance = distances[pairIndex];

        double newDistance = this.model.sample(distances, pairIndex, this.random, 1.0);
        double logDensityOld = this.model.logDensity(distances, pairIndex, oldDistance, 1.0);
        double logDensityNew = this.model.logDensity(distances, pairIndex, newDistance, 1.0);

        if (!Double.isFinite(logDensityOld) || !Double.isFinite(logDensityNew)) {
            return Double.NEGATIVE_INFINITY;
        }

        double logFactor = TreeUtils.changeNodeDistance(nodeA, nodeB, newDistance, this.random);
        return logDensityOld - logDensityNew - logFactor;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    @Override
    public double getCoercableParameterValue() {
        return scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        scaleFactor = value;
    }

    @Override
    public void optimize(double logAlpha) {
        if (2*START_TRAINING < this.count) {
            double delta = this.calcDelta(logAlpha);
            delta += Math.log(this.scaleFactor);
            this.scaleFactor = Math.exp(delta);
        }
    }

    private double[] currentDistances() {
        double[] distances = new double[this.taxonPairs.length];
        for (int i = 0; i < this.taxonPairs.length; i++) {
            TaxonPair pair = this.taxonPairs[i];
            distances[i] = getDistance(
                    this.tree.getNode(pair.firstTaxon),
                    this.tree.getNode(pair.secondTaxon));
        }
        return distances;
    }

    private double proposeAlternativeOperator() {
        if (this.alternativeOperators.isEmpty()) {
            return 0.0;
        }

        return this.alternativeOperators.get(Randomizer.nextInt(this.alternativeOperators.size())).proposal();
    }

    private static double getDistance(Node nodeA, Node nodeB) {
        Node mrca = TreeUtils.getCommonAncestor(nodeA, nodeB).mrca();
        return 2.0 * mrca.getHeight() - nodeA.getHeight() - nodeB.getHeight();
    }

    private static TaxonPair[] sampleTaxonPairs(int leafCount, int numberOfPairs) {
        List<TaxonPair> allPairs = new ArrayList<>();
        for (int i = 0; i < leafCount; i++) {
            for (int j = i + 1; j < leafCount; j++) {
                allPairs.add(new TaxonPair(i, j));
            }
        }

        int sampledPairCount = Math.min(numberOfPairs, allPairs.size());
        TaxonPair[] sampledPairs = new TaxonPair[sampledPairCount];
        int[] shuffledPairIndices = Randomizer.shuffled(allPairs.size());
        for (int i = 0; i < sampledPairCount; i++) {
            sampledPairs[i] = allPairs.get(shuffledPairIndices[i]);
        }

        return sampledPairs;
    }

    private record TaxonPair(int firstTaxon, int secondTaxon) {
    }

}
