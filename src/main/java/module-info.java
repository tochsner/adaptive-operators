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
import mala.CubeMAPGuidedMALAOperator;
import adapters.NodeValueAdapter;
import adaptiveoperators.TaxaDistanceOperator;
import adaptiveoperators.TripletDistanceOperator;
import adapters.TaxaDistanceAdapterGenerator;
import adapters.CubeAdapter;
import transforms.RealVectorIdentityTransform;
import transforms.IntVectorIdentityTransform;
import transforms.RealScalarSigmoidTransform;
import transforms.RealScalarLogTransform;
import transforms.RealVectorLogTransform;
import transforms.SimplexTransform;
import slice.StepOutShrinkSliceOperator;
import slice.LinCombSliceOperator;
import slice.MultivariateStepOutShrinkSliceOperator;
import mala.MAPGuidedMALAOperator;
import mcmc.SliceMCMC;

open module adaptiveoperators {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;
    requires org.apache.commons.math4.legacy.exception;
    requires beagle;
    requires org.apache.commons.math4.legacy.core;

    exports adaptiveoperators;
    exports weightoptimization;

    provides beast.base.core.BEASTInterface with
            AdaptiveOperator,
            BasicAdapter,
            TaxaDistanceAdapterGenerator,
            DualAveragingOperatorSchedule,
            StepOutShrinkSliceOperator,
            MultivariateStepOutShrinkSliceOperator,
            MAPGuidedMALAOperator,
            CubeAdapter,
            SliceMCMC,
            TreeTripletAdapter,
            LinCombSliceOperator,
            CubeMAPGuidedMALAOperator,
            LocalTreeAdapter,
            TreeHeightAdapter,
            MutableTreeHeightAdapter,
            NodeValueAdapter,
            NodePositionAdapter,
            TaxaDistanceOperator,
            TripletDistanceOperator,
            RealVectorIdentityTransform,
            RealScalarSigmoidTransform,
            IntVectorIdentityTransform,
            RealScalarLogTransform,
            SimplexTransform,
            RealVectorLogTransform,
            AdaptiveWeightOperator,
            RunningAverageScheme;
}
