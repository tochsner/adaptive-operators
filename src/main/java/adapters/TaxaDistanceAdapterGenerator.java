package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Tree;
import beast.base.spec.evolution.tree.ClusterTree;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

public class TaxaDistanceAdapterGenerator extends BEASTObject implements AdapterGenerator<MAPAdapter> {

    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Alignment> alignmentInput = new Input<>("alignment", "");
    public final Input<Integer> numberOfPairsInput = new Input<>("numberOfPairs",
            "number of taxon triplets to sample", 100);

    private Tree tree;
    private int numberOfPairs;
    private ClusterTree mapTree;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.numberOfPairs = this.numberOfPairsInput.get();
        this.initMapTree();
    }

    private void initMapTree() {
        this.mapTree = new ClusterTree();
        this.mapTree.clusterTypeInput.setTypedValue(ClusterTree.Type.upgma, this.mapTree);
        this.mapTree.dataInput.setTypedValue(this.alignmentInput.get(), this.mapTree);
        this.mapTree.initAndValidate();
    }

    @Override
    public List<MAPAdapter> getAdapters() {
        List<MAPAdapter> taxaDistanceAdapters = new ArrayList<>();

        for (int i = 0; i < this.numberOfPairs; i++) {
            int[] shuffledTaxa = Randomizer.shuffled(this.tree.getLeafNodeCount());
            int[] taxa = new int[] {shuffledTaxa[0], shuffledTaxa[1]};
            taxaDistanceAdapters.add(new TaxaDistanceAdapter(this.tree, taxa, this.mapTree));
        }

        return taxaDistanceAdapters;
    }
}
