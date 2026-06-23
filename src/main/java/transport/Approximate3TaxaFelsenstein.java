package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.sitemodel.SiteModelInterface;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Approximate3TaxaFelsenstein {

    private final List<String> taxonIds;
    private final TaxonSet taxonSet;
    private final int[] alignmentTaxonIndices;
    private final Alignment alignment;
    private final SiteModelInterface.Base siteModel;
    private final RealScalarParam<?> clockRate;

    public Approximate3TaxaFelsenstein(List<String> taxonIds, Alignment alignment, SiteModelInterface siteModel, RealScalarParam<?> clockRate) {
        if (taxonIds.size() != 3) {
            throw new IllegalArgumentException("Approximate3TaxaFelsenstein requires exactly three taxa");
        }
        if (!(siteModel instanceof SiteModelInterface.Base)) {
            throw new IllegalArgumentException("siteModel must be an instance of SiteModelInterface.Base");
        }

        this.taxonIds = taxonIds;
        this.alignment = alignment;
        this.siteModel = (SiteModelInterface.Base) siteModel;
        this.siteModel.setDataType(alignment.getDataType());
        if (!this.siteModel.integrateAcrossCategories()) {
            throw new IllegalArgumentException("site models with fixed site categories are not supported");
        }
        this.clockRate = Objects.requireNonNullElse(clockRate, new RealScalarParam<>(1.0, PositiveReal.INSTANCE));
        this.alignmentTaxonIndices = new int[taxonIds.size()];

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
        if (distances.length != 2) {
            throw new IllegalArgumentException("three-taxon topology requires exactly two adjacent distances");
        }

        double distanceAB = distances[0];
        double distanceBC = distances[1];

        return distanceAB <= distanceBC ? "((AB), C)" : "(A, (BC))";
    }

    private double getLogLikelihood(String topology, double[] distances) {
        double logLikelihood = 0.0;
        double clockRate = this.clockRate.get();

        for (int pattern = 0; pattern < this.alignment.getPatternCount(); pattern++) {
            double patternLikelihood = 0.0;
            patternLikelihood += this.getPatternLikelihood(topology, distances, pattern, clockRate);

            if (!Double.isFinite(patternLikelihood) || patternLikelihood <= 0.0) {
                return Double.NEGATIVE_INFINITY;
            }

            logLikelihood += Math.log(patternLikelihood) * this.alignment.getPatternWeight(pattern);
        }

        return logLikelihood;
    }

    private double getPatternLikelihood(String topology, double[] distances, int pattern, double clockRate) {
        double[] rootPartials;

        if (topology.equals("((AB), C)")) {
            rootPartials = this.getNestedLeftPartials(0, 1, 2, distances[0], distances[1], pattern, clockRate);
        } else if (topology.equals("(A, (BC))")) {
            rootPartials = this.getNestedRightPartials(0, 1, 2, distances[1], distances[0], pattern, clockRate);
        } else {
            throw new IllegalArgumentException("unknown three-taxon topology: " + topology);
        }

        double likelihood = 0.0;
        for (double rootPartial : rootPartials) {
            likelihood += 0.25 * rootPartial;
        }
        return likelihood;
    }

    private double[] getNestedLeftPartials(
            int cherryTaxonA,
            int cherryTaxonB,
            int sisterTaxon,
            double cherryDistance,
            double rootDistance,
            int pattern,
            double clockRate
    ) {
        double cherryHeight = 0.5 * cherryDistance;
        double rootHeight = 0.5 * rootDistance;

        double[] cherryPartials = this.combinePartials(
                this.getLeafPartials(cherryTaxonA, pattern), cherryHeight,
                this.getLeafPartials(cherryTaxonB, pattern), cherryHeight,
                clockRate
        );
        return this.combinePartials(
                cherryPartials, rootHeight - cherryHeight,
                this.getLeafPartials(sisterTaxon, pattern), rootHeight,
                clockRate
        );
    }

    private double[] getNestedRightPartials(
            int outgroupTaxon,
            int cherryTaxonA,
            int cherryTaxonB,
            double cherryDistance,
            double rootDistance,
            int pattern,
            double clockRate
    ) {
        double cherryHeight = 0.5 * cherryDistance;
        double rootHeight = 0.5 * rootDistance;

        double[] cherryPartials = this.combinePartials(
                this.getLeafPartials(cherryTaxonA, pattern), cherryHeight,
                this.getLeafPartials(cherryTaxonB, pattern), cherryHeight,
                clockRate
        );
        return this.combinePartials(
                this.getLeafPartials(outgroupTaxon, pattern), rootHeight,
                cherryPartials, rootHeight - cherryHeight,
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
            throw new IllegalArgumentException("JC69 three-taxon likelihood requires nucleotide data with four states");
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

    private double getJC69SameProbability(double branchLength, double effectiveRate) {
        double transition = fastExp(-4.0 * effectiveRate * branchLength / 3.0);
        return 0.25 + 0.75 * transition;
    }

    private double getJC69DifferentProbability(double branchLength, double effectiveRate) {
        double transition = fastExp(-4.0 * effectiveRate * branchLength / 3.0);
        return 0.25 - 0.25 * transition;
    }

    public static double fastExp(double val) {
        final long tmp = (long) (1512775 * val + 1072632447);
        return Double.longBitsToDouble(tmp << 32);
    }

}
