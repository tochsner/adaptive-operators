import weightoptimization.AdaptiveWeightOperator;
import weightoptimization.RunningAverageScheme;
import adaptiveoperators.AdaptiveOperator;
import adapters.BasicAdapter;
import adapters.TreeTripletAdapter;
import transforms.RealVectorIdentityTransform;
import transforms.RealScalarLogTransform;

open module adaptiveoperators {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;

    exports adaptiveoperators;
    exports weightoptimization;

    provides beast.base.core.BEASTInterface with
            AdaptiveOperator,
            BasicAdapter,
            TreeTripletAdapter,
            RealVectorIdentityTransform,
            RealScalarLogTransform,
            AdaptiveWeightOperator,
            RunningAverageScheme;
}
