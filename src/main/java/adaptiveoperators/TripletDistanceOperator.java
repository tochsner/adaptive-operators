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

public class TripletDistanceOperator extends TreeOperator {

    public final Input<Integer> numberOfTripletsInput = new Input<>(
            "numberOfTriplets",
            "number of ordered taxon triplets to learn distance proposals for",
            100);

    public final Input<List<Operator>> alternativeOperatorsInput = new Input<>(
            "alternativeOperator",
            "",
            new ArrayList<>());

    private static final int BURN_IN = 20_000;
    private static final int START_TRAINING = 60_000;
    private static final int END_TRAINING = 600_000;
    private static final double MIN_VARIANCE = 1e-12;

    private Tree tree;
    private List<Operator> alternativeOperators;

    private TaxonTriplet[] taxonTriplets;
    private NormalModel[] models;
    private Random random;
    private int count = 0;

    double scaleFactor = 1.0;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alternativeOperators = this.alternativeOperatorsInput.get();

        int leafCount = this.tree.getLeafNodeCount();
        int numberOfTriplets = this.numberOfTripletsInput.get();

        if (leafCount < 3) {
            throw new IllegalArgumentException("TripletDistanceOperator requires at least three taxa");
        }

        if (numberOfTriplets < 1) {
            throw new IllegalArgumentException("numberOfTriplets must be at least one");
        }

        this.taxonTriplets = sampleTriplets(leafCount, numberOfTriplets);
        this.models = new NormalModel[this.taxonTriplets.length];
        for (int i = 0; i < this.models.length; i++) {
            this.models[i] = new NormalModel();
        }
        this.random = new Random(Randomizer.nextLong());
    }

    @Override
    public double proposal() {
        this.count++;

        if (this.count < BURN_IN) {
            return proposeAlternativeOperator();
        }

        if (this.count < END_TRAINING) {
            recordDistances();
        }

        if (this.count < START_TRAINING) {
            return proposeAlternativeOperator();
        } else if (this.count == START_TRAINING) {
            System.out.println("Adaptive triplet tree starts");
        } else if (this.count == END_TRAINING) {
            System.out.println("Adaptive triplet tree ends");
        }

        int tripletIndex = Randomizer.nextInt(this.taxonTriplets.length);
        TaxonTriplet triplet = this.taxonTriplets[tripletIndex];
        NormalModel model = this.models[tripletIndex];

        Node nodeA = this.tree.getNode(triplet.firstTaxon);
        Node nodeB = this.tree.getNode(triplet.secondTaxon);
        Node nodeC = this.tree.getNode(triplet.thirdTaxon);
        double oldDistance = getTripletDistance(nodeA, nodeB, nodeC);

        double newDistance = model.sample(this.random, this.scaleFactor);
        double logDensityOld = model.logDensity(oldDistance, this.scaleFactor);
        double logDensityNew = model.logDensity(newDistance, this.scaleFactor);

        if (!Double.isFinite(logDensityOld) || !Double.isFinite(logDensityNew)) {
            return Double.NEGATIVE_INFINITY;
        }

        double referenceHeight = Math.min(
                getCommonAncestorHeight(nodeA, nodeC),
                getCommonAncestorHeight(nodeB, nodeC));
        double targetMrcaABHeight = referenceHeight - newDistance;
        double newABDistance = 2.0 * targetMrcaABHeight - nodeA.getHeight() - nodeB.getHeight();

        if (!Double.isFinite(newABDistance) || newABDistance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        double logFactor = TreeUtils.changeNodeDistance(nodeA, nodeB, newABDistance, this.random);
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
        if (2 * START_TRAINING < this.count) {
            double delta = this.calcDelta(logAlpha);
            delta += Math.log(this.scaleFactor);
            this.scaleFactor = Math.exp(delta);
        }
    }

    private double proposeAlternativeOperator() {
        if (this.alternativeOperators.isEmpty()) {
            return 0.0;
        }

        return this.alternativeOperators.get(Randomizer.nextInt(this.alternativeOperators.size())).proposal();
    }

    private void recordDistances() {
        for (int i = 0; i < this.taxonTriplets.length; i++) {
            TaxonTriplet triplet = this.taxonTriplets[i];
            this.models[i].record(getTripletDistance(
                    this.tree.getNode(triplet.firstTaxon),
                    this.tree.getNode(triplet.secondTaxon),
                    this.tree.getNode(triplet.thirdTaxon)));
        }
    }

    private static double getTripletDistance(Node nodeA, Node nodeB, Node nodeC) {
        double mrcaAB = getCommonAncestorHeight(nodeA, nodeB);
        double mrcaAC = getCommonAncestorHeight(nodeA, nodeC);
        double mrcaBC = getCommonAncestorHeight(nodeB, nodeC);

        return Math.min(mrcaAC, mrcaBC) - mrcaAB;
    }

    private static double getCommonAncestorHeight(Node nodeA, Node nodeB) {
        return TreeUtils.getCommonAncestor(nodeA, nodeB).mrca().getHeight();
    }

    private static TaxonTriplet[] sampleTriplets(int taxonCount, int numberOfTriplets) {
        TaxonTriplet[] triplets = new TaxonTriplet[numberOfTriplets];

        for (int i = 0; i < numberOfTriplets; i++) {
            int[] shuffledTaxa = Randomizer.shuffled(taxonCount);
            triplets[i] = new TaxonTriplet(shuffledTaxa[0], shuffledTaxa[1], shuffledTaxa[2]);
        }

        return triplets;
    }

    private record TaxonTriplet(int firstTaxon, int secondTaxon, int thirdTaxon) {
    }

    private static class NormalModel {
        private int count = 0;
        private double meanDistance = 0.0;
        private double m2Distance = 0.0;

        private void record(double distance) {
            if (!Double.isFinite(distance)) {
                return;
            }

            this.count++;
            double delta = distance - this.meanDistance;
            this.meanDistance += delta / this.count;
            double delta2 = distance - this.meanDistance;
            this.m2Distance += delta * delta2;
        }

        private double sample(Random random, double scaleFactor) {
            double standardDeviation = Math.sqrt(getVariance(scaleFactor));
            return this.meanDistance + standardDeviation * random.nextGaussian();
        }

        private double logDensity(double distance, double scaleFactor) {
            if (!Double.isFinite(distance)) {
                return Double.NEGATIVE_INFINITY;
            }

            double variance = getVariance(scaleFactor);
            double diff = distance - this.meanDistance;
            return -0.5 * (Math.log(2.0 * Math.PI * variance) + diff * diff / variance);
        }

        private double getVariance(double scaleFactor) {
            if (this.count < 2) {
                throw new IllegalStateException("At least two distances are required to sample a normal model");
            }

            return scaleFactor * Math.max(this.m2Distance / (this.count - 1), MIN_VARIANCE);
        }
    }

}
