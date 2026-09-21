package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.State;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/** A ten taxon Jukes-Cantor model to run the transport operator against in tests. */
record SmallModel(Tree tree, CompoundDistribution posterior, State state) {

    private static final String NEWICK = "((((t0:0.1,t1:0.1):0.1,t2:0.2):0.2,(t3:0.15,t4:0.15):0.25):0.3,"
            + "(((t5:0.05,t6:0.05):0.2,t7:0.25):0.2,(t8:0.3,t9:0.3):0.15):0.25);";

    static SmallModel create(Alignment data) {
        TreeParser tree = new TreeParser();
        tree.initByName("taxa", data, "newick", NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);

        JukesCantor substitutionModel = new JukesCantor();
        substitutionModel.initAndValidate();
        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", substitutionModel);

        TreeLikelihood likelihood = new TreeLikelihood();
        likelihood.initByName("data", data, "tree", tree, "siteModel", siteModel);

        // wrapped like in a real model: a compound distribution only recalculates children marked dirty,
        // so a recovery that marks too little shows up as a stale value
        CompoundDistribution posterior = new CompoundDistribution();
        posterior.initByName("distribution", likelihood);

        State state = new State();
        state.initByName("stateNode", tree);
        state.initialise();
        state.setPosterior(posterior);
        state.robustlyCalcPosterior(posterior);

        return new SmallModel(tree, posterior, state);
    }

    /** Sequences mutated from a shared ancestor, so that pairwise distances stay finite. */
    static Alignment createAlignment() {
        Random random = new Random(1);
        char[] bases = {'A', 'C', 'G', 'T'};
        char[] ancestor = new char[300];
        for (int site = 0; site < ancestor.length; site++) {
            ancestor[site] = bases[random.nextInt(4)];
        }

        List<Sequence> sequences = new ArrayList<>();
        for (int taxon = 0; taxon < 10; taxon++) {
            char[] sequence = ancestor.clone();
            for (int site = 0; site < sequence.length; site++) {
                if (random.nextDouble() < 0.05 + 0.02 * taxon) {
                    sequence[site] = bases[random.nextInt(4)];
                }
            }
            sequences.add(new Sequence("t" + taxon, new String(sequence)));
        }

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");
        return alignment;
    }

    /** The clades of the tree, which identify its topology. */
    Set<BitSet> getClades() {
        Set<BitSet> clades = new HashSet<>();

        for (Node node : this.tree.getInternalNodes()) {
            BitSet clade = new BitSet();
            for (Node leaf : node.getAllLeafNodes()) {
                clade.set(leaf.getNr());
            }
            clades.add(clade);
        }

        return clades;
    }

    /** Recalculates the posterior the way JumpMCMC does after a rejected proposal. */
    double recoverAfterRejection(GuidedNodeTransportOperator operator) {
        this.state.restore();
        operator.markDirtyAfterRestore(this.state);
        this.state.checkCalculationNodesDirtiness();
        double logP = this.posterior.calculateLogP();
        this.state.setEverythingDirty(false);
        this.state.acceptCalculationNodes();

        return logP;
    }
}
