package transport;

import adapters.TreeUtils;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.distance.Distance;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.tree.ClusterTree;
import beast.base.spec.inference.parameter.RealScalarParam;

import java.util.*;

public class ApproximateFelsenstein {

    private final List<String> taxonIds;
    private final TaxonSet taxonSet;
    private Tree tree;
    private final int[] alignmentTaxonIndices;
    LinkedList<Integer> cube;
    Alignment alignment;
    RealScalarParam<?> clockRate;

    public ApproximateFelsenstein(List<String> taxonIds, Alignment alignment, RealScalarParam<?> clockRate) {
        this.taxonIds = taxonIds;
        this.alignment = alignment;
        this.clockRate = Objects.requireNonNullElse(clockRate, new RealScalarParam<>(1.0, PositiveReal.INSTANCE));
        this.alignmentTaxonIndices = new int[taxonIds.size()];

        this.cube = new LinkedList<>();
        for (int i = 0; i < this.taxonIds.size(); i++) {
            cube.add(i);
        }

        List<Taxon> taxa = new ArrayList<>();
        for (int i = 0; i < taxonIds.size(); i++) {
            String taxonId = taxonIds.get(i);
            taxa.add(new Taxon(taxonId));

            int taxonIndex = this.alignment.getTaxonIndex(taxonId);
            if (taxonIndex < 0) {
                throw new IllegalArgumentException("taxon " + taxonId + " not found in alignment");
            }
            this.alignmentTaxonIndices[i] = taxonIndex;
        }

        this.taxonSet = new TaxonSet();
        this.taxonSet.taxonsetInput.setTypedValue(taxa, this.taxonSet);
        this.taxonSet.initAndValidate();
    }

    public double getApproximateLogFelsenstein(double[] distances) {
        Tree tree = this.buildTree(distances);
        double clockRate = this.clockRate.get();

        double logLikelihood = 0.0;
        for (int pattern = 0; pattern < this.alignment.getPatternCount(); pattern++) {
            double patternLikelihood = this.getPatternLikelihood(tree.getRoot(), pattern, clockRate);
            if (!Double.isFinite(patternLikelihood) || patternLikelihood <= 0.0) {
                return 0.0;
            }

            logLikelihood += Math.log(patternLikelihood) * this.alignment.getPatternWeight(pattern);
        }

        return logLikelihood;
    }

    public Tree buildTree(double[] distances) {
        if (this.tree == null) {

            int n = this.taxonIds.size();

            double [] distanceMatrix = new double[n * n];
            Arrays.fill(distanceMatrix, Double.MAX_VALUE);

            for (int i = 0; i < distances.length; i++) {
                int a = i;
                int b = i + 1;

                distanceMatrix[a * n + b] = distances[i];
                distanceMatrix[a + b * n] = distances[i];
            }

            Distance distance = (taxon1, taxon2) -> distanceMatrix[taxon1 * n + taxon2];

            this.tree = new ClusterTree();
            this.tree.initByName(
                    "clusterType", "single",
                    "distance", distance,
                    "taxonset", this.taxonSet
            );

        } else {

            TreeUtils.setCubeDistances(this.tree, cube, distances);

        }

        return this.tree;
    }

    private double getPatternLikelihood(Node root, int pattern, double clockRate) {
        double[] partials = this.getPartials(root, pattern, clockRate);

        double likelihood = 0.0;
        for (double partial : partials) {
            likelihood += 0.25 * partial;
        }
        return likelihood;
    }

    private double[] getPartials(Node node, int pattern, double clockRate) {
        int stateCount = this.alignment.getDataType().getStateCount();

        if (node.isLeaf()) {
            return this.getLeafPartials(node, pattern, stateCount);
        }

        double[] leftPartials = this.getPartials(node.getLeft(), pattern, clockRate);
        double[] rightPartials = this.getPartials(node.getRight(), pattern, clockRate);
        double[] partials = new double[stateCount];

        double leftBranchLength = node.getHeight() - node.getLeft().getHeight();
        double rightBranchLength = node.getHeight() - node.getRight().getHeight();

        for (int state = 0; state < stateCount; state++) {
            double leftContribution = this.getBranchContribution(state, leftPartials, leftBranchLength, clockRate);
            double rightContribution = this.getBranchContribution(state, rightPartials, rightBranchLength, clockRate);
            partials[state] = leftContribution * rightContribution;
        }

        return partials;
    }

    private double[] getLeafPartials(Node node, int pattern, int stateCount) {
        String taxonId = node.getID();
        int localTaxonIndex = this.taxonIds.indexOf(taxonId);
        if (localTaxonIndex < 0) {
            throw new IllegalArgumentException("taxon " + taxonId + " not found in local taxa");
        }

        int alignmentTaxonIndex = this.alignmentTaxonIndices[localTaxonIndex];
        int stateCode = this.alignment.getPattern(alignmentTaxonIndex, pattern);
        boolean[] stateSet = this.alignment.getDataType().getStateSet(stateCode);

        double[] partials = new double[stateCount];
        for (int state = 0; state < stateCount; state++) {
            partials[state] = stateSet[state] ? 1.0 : 0.0;
        }
        return partials;
    }

    private double getBranchContribution(int parentState, double[] childPartials, double branchLength, double clockRate) {
        double pSame = this.getJC69SameProbability(branchLength, clockRate);
        double pDifferent = this.getJC69DifferentProbability(branchLength, clockRate);

        double contribution = 0.0;
        for (int childState = 0; childState < childPartials.length; childState++) {
            double transitionProbability = parentState == childState ? pSame : pDifferent;
            contribution += transitionProbability * childPartials[childState];
        }
        return contribution;
    }

    private double getJC69SameProbability(double branchLength, double clockRate) {
        double transition = Math.exp(-4.0 * clockRate * branchLength / 3.0);
        return 0.25 + 0.75 * transition;
    }

    private double getJC69DifferentProbability(double branchLength, double clockRate) {
        double transition = Math.exp(-4.0 * clockRate * branchLength / 3.0);
        return 0.25 - 0.25 * transition;
    }

    public static double fastExp(double val) {
        final long tmp = (long) (1512775 * val + 1072632447);
        return Double.longBitsToDouble(tmp << 32);
    }

}
