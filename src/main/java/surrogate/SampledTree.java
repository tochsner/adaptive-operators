package surrogate;

import beast.base.evolution.tree.Tree;

public record SampledTree(Tree topology, double logLikelihood) {
}
