import weightoptimization.AdaptiveWeightOperator;
import weightoptimization.ESSWeightAdaptationOperator;
import adapters.RobinsonFouldsDistanceAdapter;
import mcmc.OperatorTimingMCMC;
import weightoptimization.RunningAverageScheme;
import schedule.DualAveragingOperatorSchedule;
import schedule.ESSWeightOperatorSchedule;
import adaptiveoperators.AdaptiveOperator;
import adapters.BasicAdapter;
import adapters.TreeTripletAdapter;
import adapters.LocalTreeAdapter;
import adapters.NodeEmbeddingAdapter;
import adapters.TreeHeightAdapter;
import adapters.TreeLengthAdapter;
import adapters.MutableTreeHeightAdapter;
import adapters.TreeDescriptionAdapter;
import adapters.NodePositionAdapter;
import adapters.Adapter;
import mala.CubeMAPGuidedMALAOperator;
import adapters.NodeValueAdapter;
import adaptiveoperators.TaxaDistanceOperator;
import adaptiveoperators.TripletDistanceOperator;
import adapters.TaxaDistanceAdapterGenerator;
import adapters.CubeAdapter;
import transforms.RealVectorIdentityTransform;
import transforms.RealScalarIdentityTransform;
import transforms.IntVectorIdentityTransform;
import transforms.RealScalarSigmoidTransform;
import transforms.RealScalarLogTransform;
import transforms.RealVectorLogTransform;
import transforms.SimplexTransform;
import transforms.IntSimplexRoundTransform;
import transforms.SimplexRoundTransform;
import transport.PartialCubeTransportOperator;
import transport.NodeTransportOperator;
import transport.GuidedNodeTransportOperator;
import slice.StepOutShrinkSliceOperator;
import slice.LinCombSliceOperator;
import multipletry.MultipleTryOperator;
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
import mcmc.OperatorTimingMCMC;
import mcmc.JumpMCMC;
import mcmc.ESSStoppingMCMC;
import surrogate.SimulatedAnnealing;
import surrogate.SATreeGraphOperator;
import surrogate.TreeGraphOperator;
import upgma.UpgmaRefreshOperator;
import delayedacceptance.ApproximateDelayedAcceptanceMCMC;
import delayedacceptance.DelayedAcceptanceMCMC;
import delayedacceptance.MLPPosteriorApproximation;
import distancematrix.DistanceMatrixOperator;
import operatorsleection.OrderedOperatorMCMC;

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
    exports adapters;
    exports mala;

    provides beast.base.core.BEASTInterface with
            DistanceMatrixOperator,
            OrderedOperatorMCMC,
            AdaptiveOperator,
            TreeDescriptionAdapter,
            SATreeGraphOperator,
            TreeGraphOperator,
            BasicAdapter,
            TaxaDistanceAdapterGenerator,
            NodeEmbeddingAdapter,
            IntSimplexRoundTransform,
            RealScalarIdentityTransform,
            SimulatedAnnealing,
            DualAveragingOperatorSchedule,
            ESSWeightOperatorSchedule,
            StepOutShrinkSliceOperator,
            MultivariateStepOutShrinkSliceOperator,
            MAPGuidedMALAOperator,
            UpgmaRefreshOperator,
            CubeAdapter,
            MalaMCMC,
            OperatorTimingMCMC,
            SliceMCMC,
            TreeTripletAdapter,
            GuidedRandomWalkOperator,
            MALAOperator,
            LinCombSliceOperator,
            MultipleTryOperator,
            GuidedNodeTransportOperator,
            CubeMAPGuidedMALAOperator,
            FisherMALAOperator,
            LargeJumpMALAOperator,
            LocalTreeAdapter,
            TreeHeightAdapter,
            TreeLengthAdapter,
            MutableTreeHeightAdapter,
            NodeValueAdapter,
            JumpMCMC,
            ESSStoppingMCMC,
            NodePositionAdapter,
            MLMCMC,
            NodeTransportOperator,
            UnifiedLargeJumpMALAOperator,
            ApproximateDelayedAcceptanceMCMC,
            DelayedAcceptanceMCMC,
            MLPPosteriorApproximation,
            TaxaDistanceOperator,
            TripletDistanceOperator,
            RealVectorIdentityTransform,
            RealScalarSigmoidTransform,
            PartialCubeTransportOperator,
            IntVectorIdentityTransform,
            RealScalarLogTransform,
            SimplexTransform,
            PreconditionedCrankNicolsonOperator,
            GuidedMixedPreconditionedCrankNicolsonOperator,
            SimplexRoundTransform,
            RealVectorLogTransform,
            AdaptiveWeightOperator,
            ESSWeightAdaptationOperator,
            RobinsonFouldsDistanceAdapter,
            RunningAverageScheme;
}
