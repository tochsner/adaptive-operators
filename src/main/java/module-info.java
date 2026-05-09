import weightoptimization.AdaptiveWeightOperator;
import weightoptimization.RunningAverageScheme;
import adaptiveoperators.MyScaleOperator;

open module adaptiveoperators {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;

    exports adaptiveoperators;
    exports weightoptimization;

    provides beast.base.core.BEASTInterface with
            MyScaleOperator,
            AdaptiveWeightOperator,
            RunningAverageScheme;
}
