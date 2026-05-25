import weightoptimization.AdaptiveWeightOperator;
import weightoptimization.RunningAverageScheme;
import schedule.DualAveragingOperatorSchedule;
import adaptiveoperators.AdaptiveOperator;
import adapters.BasicAdapter;
import adapters.TreeTripletAdapter;
import adapters.LocalTreeAdapter;
import adapters.TreeHeightAdapter;
import adapters.MutableTreeHeightAdapter;
import adapters.NodePositionAdapter;
import adapters.NodeValueAdapter;
import adaptiveoperators.TaxaDistanceOperator;
import adaptiveoperators.TripletDistanceOperator;
import adapters.TaxaDistanceAdapterGenerator;
import transforms.RealVectorIdentityTransform;
import transforms.RealScalarSigmoidTransform;
import transforms.IntVectorIdentityTransform;
import transforms.RealScalarLogTransform;
import transforms.RealVectorLogTransform;
import transforms.SimplexTransform;
import slice.StepOutShrinkSliceOperator;
import slice.MultivariateStepOutShrinkSliceOperator;
import mcmc.SliceMCMC;

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
            TaxaDistanceAdapterGenerator,
            DualAveragingOperatorSchedule,
            StepOutShrinkSliceOperator,
            MultivariateStepOutShrinkSliceOperator,
            SliceMCMC,
            RealScalarSigmoidTransform,
            TreeTripletAdapter,
            LocalTreeAdapter,
            TreeHeightAdapter,
            MutableTreeHeightAdapter,
            NodeValueAdapter,
            NodePositionAdapter,
            TaxaDistanceOperator,
            TripletDistanceOperator,
            RealVectorIdentityTransform,
            IntVectorIdentityTransform,
            RealScalarLogTransform,
            SimplexTransform,
            RealVectorLogTransform,
            AdaptiveWeightOperator,
            RunningAverageScheme;
}
