import weightoptimization.AdaptiveWeightOperator;
import weightoptimization.RunningAverageScheme;
import schedule.DualAveragingOperatorSchedule;
import adaptiveoperators.AdaptiveOperator;
import adapters.BasicAdapter;
import adapters.TreeTripletAdapter;
import adapters.LocalTreeAdapter;
import adapters.TreeHeightAdapter;
import adapters.NodePositionAdapter;
import adapters.NodeValueAdapter;
import adaptiveoperators.TaxaDistanceOperator;
import transforms.RealVectorIdentityTransform;
import transforms.IntVectorIdentityTransform;
import transforms.RealScalarLogTransform;
import transforms.RealVectorLogTransform;
import transforms.SimplexTransform;

open module adaptiveoperators {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;
    requires org.apache.commons.math4.legacy.exception;
    requires beagle;

    exports adaptiveoperators;
    exports weightoptimization;

    provides beast.base.core.BEASTInterface with
            AdaptiveOperator,
            BasicAdapter,
            DualAveragingOperatorSchedule,
            TreeTripletAdapter,
            LocalTreeAdapter,
            TreeHeightAdapter,
            NodeValueAdapter,
            NodePositionAdapter,
            TaxaDistanceOperator,
            RealVectorIdentityTransform,
            IntVectorIdentityTransform,
            RealScalarLogTransform,
            SimplexTransform,
            RealVectorLogTransform,
            AdaptiveWeightOperator,
            RunningAverageScheme;
}
