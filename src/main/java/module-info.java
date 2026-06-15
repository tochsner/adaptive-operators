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
import transforms.IntSimplexRoundTransform;
import transforms.SimplexRoundTransform;
import slice.StepOutShrinkSliceOperator;
import slice.LinCombSliceOperator;
import slice.MultivariateStepOutShrinkSliceOperator;
import mala.MAPGuidedMALAOperator;
import mala.MALAOperator;
import adaptiveoperators.PreconditionedCrankNicolsonOperator;
import adaptiveoperators.GuidedMixedPreconditionedCrankNicolsonOperator;
import mala.FisherMALAOperator;
import irreversible.GuidedRandomWalkOperator;
import largejump.LargeJumpMALAOperator;
import largejump.UnifiedLargeJumpMALAOperator;
import mcmc.SliceMCMC;
import mcmc.MLMCMC;
import mcmc.MalaMCMC;
import mcmc.JumpMCMC;
import delayedacceptance.ApproximateDelayedAcceptanceMCMC;
import delayedacceptance.DelayedAcceptanceMCMC;
import delayedacceptance.MLPPosteriorApproximation;

open module adaptiveoperators {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;
    requires org.apache.commons.math4.legacy.exception;
    requires beagle;
    requires org.apache.commons.math4.legacy.core;
    requires ai.djl.api;
    requires ai.djl.pytorch_engine;
    requires ai.djl.basicdataset;
    requires ai.djl.model_zoo;
    requires ai.djl.pytorch_model_zoo;
    requires org.slf4j;

    exports adaptiveoperators;
    exports weightoptimization;

    provides beast.base.core.BEASTInterface with
            AdaptiveOperator,
            BasicAdapter,
            TaxaDistanceAdapterGenerator,
            IntSimplexRoundTransform,
            DualAveragingOperatorSchedule,
            StepOutShrinkSliceOperator,
            MultivariateStepOutShrinkSliceOperator,
            MAPGuidedMALAOperator,
            CubeAdapter,
            MalaMCMC,
            SliceMCMC,
            TreeTripletAdapter,
            GuidedRandomWalkOperator,
            MALAOperator,
            LinCombSliceOperator,
            CubeMAPGuidedMALAOperator,
            FisherMALAOperator,
            LargeJumpMALAOperator,
            LocalTreeAdapter,
            TreeHeightAdapter,
            MutableTreeHeightAdapter,
            NodeValueAdapter,
            JumpMCMC,
            NodePositionAdapter,
            MLMCMC,
            UnifiedLargeJumpMALAOperator,
            ApproximateDelayedAcceptanceMCMC,
            DelayedAcceptanceMCMC,
            MLPPosteriorApproximation,
            TaxaDistanceOperator,
            TripletDistanceOperator,
            RealVectorIdentityTransform,
            RealScalarSigmoidTransform,
            IntVectorIdentityTransform,
            RealScalarLogTransform,
            SimplexTransform,
            PreconditionedCrankNicolsonOperator,
            GuidedMixedPreconditionedCrankNicolsonOperator,
            SimplexRoundTransform,
            RealVectorLogTransform,
            AdaptiveWeightOperator,
            RunningAverageScheme;
}
