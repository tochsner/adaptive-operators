package transport;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.distance.JukesCantorDistance;

import java.util.Arrays;

public class FastJCDistance extends JukesCantorDistance {

    private double[] cachedDistances;
    private int taxonCount;

    @Override
    public void setPatterns(Alignment patterns) {
        super.setPatterns(patterns);

        this.taxonCount = patterns.getTaxonCount();
        this.cachedDistances = new double[this.taxonCount * this.taxonCount];
        Arrays.fill(this.cachedDistances, Double.NaN);
    }

    @Override
    public double pairwiseDistance(int i, int j) {
        int index = i * this.taxonCount + j;
        double distance = this.cachedDistances[index];
        if (!Double.isNaN(distance)) {
            return distance;
        }

        distance = super.pairwiseDistance(i, j);
        this.cachedDistances[index] = distance;
        this.cachedDistances[j * this.taxonCount + i] = distance;
        return distance;
    }

}
