/*
 * JukesCantorDistanceMatrix.java
 *
 * Copyright (C) 2002-2006 Alexei Drummond and Andrew Rambaut
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

package mala;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.distance.JukesCantorDistance;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

public class SubsetJukesCantorDistance extends JukesCantorDistance {

    private final double fraction;
    List<Integer> indicesConsidered;

    public SubsetJukesCantorDistance(double fraction) {
        this.fraction = fraction;
    }

    @Override
    public void setPatterns(Alignment patterns) {
        super.setPatterns(patterns);

        final int stateCount = dataType.getStateCount();

        const1 = ((double) stateCount - 1) / stateCount;
        const2 = ((double) stateCount) / (stateCount - 1);

        this.indicesConsidered = new ArrayList<>();

        for (int i = 0; i < this.patterns.getPatternCount(); i++) {
            if (Randomizer.nextDouble() < this.fraction) {
                this.indicesConsidered.add(i);
            }
        }
    }

    @Override
    public double pairwiseDistance(int taxon1, int taxon2) {
        int state1, state2;

        double weight;
        double sumDistance = 0.0;
        double sumWeight = 0.0;

        int[] pattern;

        for (int i : this.indicesConsidered) {
            pattern = patterns.getPattern(i);

            state1 = pattern[taxon1];
            state2 = pattern[taxon2];

            weight = patterns.getPatternWeight(i);
            if (!dataType.isAmbiguousCode(state1) && !dataType.isAmbiguousCode(state2) &&
                    state1 != state2) {
                sumDistance += weight;
            }
            sumWeight += weight;
        }

        double obsDist = sumDistance / sumWeight;

        if (obsDist == 0.0) return 0.0;

        if (obsDist >= const1) {
            return MAX_DISTANCE;
        }

        final double expDist = -const1 * Math.log(1.0 - (const2 * obsDist));

        if (expDist < MAX_DISTANCE) {
            return expDist;
        } else {
            return MAX_DISTANCE;
        }
    }

    private double const1, const2;

}
