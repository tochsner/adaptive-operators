import adaptiveoperators.MyDistribution;
import adaptiveoperators.MyScaleOperator;

open module adaptiveoperators {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;

    exports adaptiveoperators;

    provides beast.base.core.BEASTInterface with
            MyDistribution,
            MyScaleOperator;
}
