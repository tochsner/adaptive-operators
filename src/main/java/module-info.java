import adaptiveoperators.MyScaleOperator;

open module adaptiveoperators {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;

    exports adaptiveoperators;

    provides beast.base.core.BEASTInterface with
            MyScaleOperator;
}
