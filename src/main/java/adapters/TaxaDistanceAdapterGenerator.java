package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Tree;
import beast.base.spec.evolution.tree.ClusterTree;
import beast.base.spec.type.RealScalar;
import beast.base.spec.type.RealVector;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

public class TaxaDistanceAdapterGenerator extends BEASTObject implements AdapterGenerator<MAPAdapter> {

    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    public final Input<RealScalar<?>> clockRateInput = new Input<>("clockRate", "");
    public final Input<RealVector<?>> clockRatesInput = new Input<>("clockRates", "");
    public final Input<Integer> numberOfPairsInput = new Input<>("numberOfPairs",
            "number of taxon triplets to sample", 100);

    private Tree tree;
    private RealVector<?> clockRates;
    private RealScalar<?> clockRate;
    private int numberOfPairs;
    private List<Tree> mapTrees;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.clockRate = this.clockRateInput.get();
        this.clockRates = this.clockRatesInput.get();
        this.numberOfPairs = this.numberOfPairsInput.get();
        this.initMapTrees();
    }

    private void initMapTrees() {
        this.mapTrees = new ArrayList<>();
        this.addTree(ClusterTree.Type.single);
        this.addTree(ClusterTree.Type.average);
        this.addTree(ClusterTree.Type.complete);
        this.addTree(ClusterTree.Type.neighborjoining2);
        this.addTree(ClusterTree.Type.upgma);
    }

    private void addTree(ClusterTree.Type type) {
        ClusterTree tree = new ClusterTree();
        tree.clusterTypeInput.setTypedValue(type, tree);
        tree.dataInput.setTypedValue(this.alignmentInput.get(), tree);
        tree.m_traitList.setTypedValue(this.tree.m_traitList.get(), tree);
        tree.initAndValidate();

        this.mapTrees.add(tree);
    }

    @Override
    public List<MAPAdapter> getAdapters() {
        List<MAPAdapter> taxaDistanceAdapters = new ArrayList<>();

        for (int i = 0; i < this.numberOfPairs; i++) {
            int[] shuffledTaxa = Randomizer.shuffled(this.tree.getLeafNodeCount());
            int[] taxa = new int[] {shuffledTaxa[0], shuffledTaxa[1]};
            taxaDistanceAdapters.add(new TaxaDistanceAdapter(this.tree, taxa, this.mapTrees, this.clockRate, this.clockRates));
        }

        return taxaDistanceAdapters;
    }
}
