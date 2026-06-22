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

public class Approximate4TaxaFelsenstein {

    private final List<String> taxonIds;
    private final TaxonSet taxonSet;
    private final int[] alignmentTaxonIndices;
    private final LinkedList<Integer> cube;
    private final Alignment alignment;
    private final RealScalarParam<?> clockRate;

    public Approximate4TaxaFelsenstein(List<String> taxonIds, Alignment alignment, RealScalarParam<?> clockRate) {
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
        String topology = this.getTopology(distances);
        return this.getLogLikelihood(topology, distances);
    }

    private String getTopology(double[] distances) {
        if (distances.length != 3) {
            throw new IllegalArgumentException("four-taxon topology requires exactly three adjacent distances");
        }

        double distanceAB = distances[0];
        double distanceBC = distances[1];
        double distanceCD = distances[2];

        if (distanceAB <= distanceBC && distanceAB <= distanceCD) {
            if (distanceBC <= distanceCD) {
                return "((AB, C), D)";
            }
            return "(AB, CD)";
        }

        if (distanceBC <= distanceAB && distanceBC <= distanceCD) {
            if (distanceAB <= distanceCD) {
                return "((A, BC), D)";
            }
            return "(A, (BC, D))";
        }

        if (distanceAB <= distanceBC) {
            return "(AB, CD)";
        }
        return "(A, (B, CD))";
    }

    private double getLogLikelihood(String topology, double[] distances) {
        double logLikelihood = 0.0;
        double clockRate = this.clockRate.get();

        for (int pattern = 0; pattern < this.alignment.getPatternCount(); pattern++) {
            double patternLikelihood = this.getPatternLikelihood(topology, distances, pattern, clockRate);
            if (!Double.isFinite(patternLikelihood) || patternLikelihood <= 0.0) {
                return Double.NEGATIVE_INFINITY;
            }

            logLikelihood += Math.log(patternLikelihood) * this.alignment.getPatternWeight(pattern);
        }

        return logLikelihood;
    }

    private double getPatternLikelihood(String topology, double[] distances, int pattern, double clockRate) {
        double[] rootPartials;

        if (topology.equals("((A, BC), D)")) {
            rootPartials = this.getNestedLeftPartials(1, 2, 0, 3, distances[1], distances[0], distances[2], pattern, clockRate);
        } else if (topology.equals("((AB, C), D)")) {
            rootPartials = this.getNestedLeftPartials(0, 1, 2, 3, distances[0], distances[1], distances[2], pattern, clockRate);
        } else if (topology.equals("(A, (B, CD))")) {
            rootPartials = this.getNestedRightPartials(0, 1, 2, 3, distances[2], distances[1], distances[0], pattern, clockRate);
        } else if (topology.equals("(A, (BC, D))")) {
            rootPartials = this.getNestedRightPartials(0, 3, 1, 2, distances[1], distances[2], distances[0], pattern, clockRate);
        } else if (topology.equals("(AB, CD)")) {
            rootPartials = this.getBalancedPartials(distances, pattern, clockRate);
        } else {
            throw new IllegalArgumentException("unknown four-taxon topology: " + topology);
        }

        double likelihood = 0.0;
        for (double partial : rootPartials) {
            likelihood += 0.25 * partial;
        }
        return likelihood;
    }

    private double[] getNestedLeftPartials(
            int cherryTaxonA,
            int cherryTaxonB,
            int sisterTaxon,
            int outgroupTaxon,
            double cherryDistance,
            double sisterDistance,
            double rootDistance,
            int pattern,
            double clockRate
    ) {
        double cherryHeight = 0.5 * cherryDistance;
        double sisterHeight = 0.5 * sisterDistance;
        double rootHeight = 0.5 * rootDistance;

        double[] cherryPartials = this.combinePartials(
                this.getLeafPartials(cherryTaxonA, pattern), cherryHeight,
                this.getLeafPartials(cherryTaxonB, pattern), cherryHeight,
                clockRate
        );
        double[] innerPartials = this.combinePartials(
                cherryPartials, sisterHeight - cherryHeight,
                this.getLeafPartials(sisterTaxon, pattern), sisterHeight,
                clockRate
        );
        return this.combinePartials(
                innerPartials, rootHeight - sisterHeight,
                this.getLeafPartials(outgroupTaxon, pattern), rootHeight,
                clockRate
        );
    }

    private double[] getNestedRightPartials(
            int outgroupTaxon,
            int sisterTaxon,
            int cherryTaxonA,
            int cherryTaxonB,
            double cherryDistance,
            double sisterDistance,
            double rootDistance,
            int pattern,
            double clockRate
    ) {
        double cherryHeight = 0.5 * cherryDistance;
        double sisterHeight = 0.5 * sisterDistance;
        double rootHeight = 0.5 * rootDistance;

        double[] cherryPartials = this.combinePartials(
                this.getLeafPartials(cherryTaxonA, pattern), cherryHeight,
                this.getLeafPartials(cherryTaxonB, pattern), cherryHeight,
                clockRate
        );
        double[] innerPartials = this.combinePartials(
                this.getLeafPartials(sisterTaxon, pattern), sisterHeight,
                cherryPartials, sisterHeight - cherryHeight,
                clockRate
        );
        return this.combinePartials(
                this.getLeafPartials(outgroupTaxon, pattern), rootHeight,
                innerPartials, rootHeight - sisterHeight,
                clockRate
        );
    }

    private double[] getBalancedPartials(double[] distances, int pattern, double clockRate) {
        double heightAB = 0.5 * distances[0];
        double rootHeight = 0.5 * distances[1];
        double heightCD = 0.5 * distances[2];

        double[] partialsAB = this.combinePartials(
                this.getLeafPartials(0, pattern), heightAB,
                this.getLeafPartials(1, pattern), heightAB,
                clockRate
        );
        double[] partialsCD = this.combinePartials(
                this.getLeafPartials(2, pattern), heightCD,
                this.getLeafPartials(3, pattern), heightCD,
                clockRate
        );
        return this.combinePartials(
                partialsAB, rootHeight - heightAB,
                partialsCD, rootHeight - heightCD,
                clockRate
        );
    }

    private double[] combinePartials(
            double[] leftPartials,
            double leftBranchLength,
            double[] rightPartials,
            double rightBranchLength,
            double clockRate
    ) {
        int stateCount = leftPartials.length;
        double[] partials = new double[stateCount];

        for (int state = 0; state < stateCount; state++) {
            double leftContribution = this.getBranchContribution(
                    state, leftPartials, Math.max(0.0, leftBranchLength), clockRate
            );
            double rightContribution = this.getBranchContribution(
                    state, rightPartials, Math.max(0.0, rightBranchLength), clockRate
            );
            partials[state] = leftContribution * rightContribution;
        }

        return partials;
    }

    private double[] getLeafPartials(int localTaxonIndex, int pattern) {
        int stateCount = this.alignment.getDataType().getStateCount();
        if (stateCount != 4) {
            throw new IllegalArgumentException("JC69 four-taxon likelihood requires nucleotide data with four states");
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

}
