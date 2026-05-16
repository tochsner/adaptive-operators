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

    private static final int BURN_IN = 20_000;
    private static final int NUM_TRAINING = 60_000;
    private static final double MIN_LOG_VARIANCE = 1e-12;

    private Tree tree;
    private List<Operator> alternativeOperators;

    private TaxonPair[] taxonPairs;
    private LogNormalModel[] models;
    private Random random;
    private int count = 0;

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
        this.models = new LogNormalModel[this.taxonPairs.length];
        for (int i = 0; i < this.models.length; i++) {
            this.models[i] = new LogNormalModel();
        }
        this.random = new Random(Randomizer.nextLong());
    }

    @Override
    public double proposal() {
        this.count++;

        if (this.count < BURN_IN) {
            return this.alternativeOperators.get(Randomizer.nextInt(this.alternativeOperators.size())).proposal();
        }

        recordDistances();

        if (this.count < NUM_TRAINING) {
            return this.alternativeOperators.get(Randomizer.nextInt(this.alternativeOperators.size())).proposal();
        } else if (this.count == NUM_TRAINING) {
            System.out.println("Adaptive tree starts");
        }

        int pairIndex = Randomizer.nextInt(this.taxonPairs.length);
        TaxonPair pair = this.taxonPairs[pairIndex];
        LogNormalModel model = this.models[pairIndex];

        Node nodeA = this.tree.getNode(pair.firstTaxon);
        Node nodeB = this.tree.getNode(pair.secondTaxon);
        double oldDistance = getDistance(nodeA, nodeB);

        try {
            double newDistance = model.sample(this.random);
            double logDensityOld = model.logDensity(oldDistance);
            double logDensityNew = model.logDensity(newDistance);

            if (!Double.isFinite(logDensityOld) || !Double.isFinite(logDensityNew)) {
                return Double.NEGATIVE_INFINITY;
            }

            TreeUtils.changeNodeDistance(nodeA, nodeB, newDistance, this.random);
            return logDensityOld - logDensityNew;
        } catch (RuntimeException e) {
            return Double.NEGATIVE_INFINITY;
        }
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

    private void recordDistances() {
        for (int i = 0; i < this.taxonPairs.length; i++) {
            TaxonPair pair = this.taxonPairs[i];
            this.models[i].record(getDistance(
                    this.tree.getNode(pair.firstTaxon),
                    this.tree.getNode(pair.secondTaxon)));
        }
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

    private static class LogNormalModel {
        private int count = 0;
        private double meanLogDistance = 0.0;
        private double m2LogDistance = 0.0;

        private void record(double distance) {
            if (!Double.isFinite(distance) || distance <= 0.0) {
                return;
            }

            double logDistance = Math.log(distance);
            this.count++;
            double delta = logDistance - this.meanLogDistance;
            this.meanLogDistance += delta / this.count;
            double delta2 = logDistance - this.meanLogDistance;
            this.m2LogDistance += delta * delta2;
        }

        private double sample(Random random) {
            double standardDeviation = Math.sqrt(getVariance());
            return Math.exp(this.meanLogDistance + standardDeviation * random.nextGaussian());
        }

        private double logDensity(double distance) {
            if (!Double.isFinite(distance) || distance <= 0.0) {
                return Double.NEGATIVE_INFINITY;
            }

            double variance = getVariance();
            double logDistance = Math.log(distance);
            double diff = logDistance - this.meanLogDistance;
            return -Math.log(distance)
                    - 0.5 * (Math.log(2.0 * Math.PI * variance) + diff * diff / variance);
        }

        private double getVariance() {
            if (this.count < 2) {
                throw new IllegalStateException("At least two distances are required to sample a log-normal model");
            }

            return Math.max(this.m2LogDistance / (this.count - 1), MIN_LOG_VARIANCE);
        }
    }

}
