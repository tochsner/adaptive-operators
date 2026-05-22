package adapters;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

public class TaxaDistanceAdapterGenerator extends BEASTObject implements AdapterGenerator {

    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Integer> numberOfPairsInput = new Input<>("numberOfPairs",
            "number of taxon triplets to sample", 30);

    private Tree tree;
    private int numberOfPairs;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.numberOfPairs = this.numberOfPairsInput.get();
    }

    private static int[][] sampleTriplets(int taxonCount, int numberOfTriplets) {
        int[][] triplets = new int[numberOfTriplets][3];

        for (int i = 0; i < numberOfTriplets; i++) {
            int[] shuffledTaxa = Randomizer.shuffled(taxonCount);
            triplets[i][0] = shuffledTaxa[0];
            triplets[i][1] = shuffledTaxa[1];
            triplets[i][2] = shuffledTaxa[2];
        }

        return triplets;
    }

    @Override
    public List<Adapter> getAdapters() {
        List<Adapter> taxaDistanceAdapters = new ArrayList<>();

        for (int i = 0; i < this.numberOfPairs; i++) {
            int[] shuffledTaxa = Randomizer.shuffled(this.tree.getLeafNodeCount());
            int[] taxa = new int[] {shuffledTaxa[0], shuffledTaxa[1]};
            taxaDistanceAdapters.add(new TaxaDistanceAdapter(this.tree, taxa));
        }

        return taxaDistanceAdapters;
    }
}
