# Adaptive Operators

This README summarizes the operator approaches implemented in this package:

- [Adapters](#adapters)
- [Learned Conditional Proposals](#learned-conditional-proposals)
- [Learned Tree-Distance Proposals](#learned-tree-distance-proposals)
- [Slice-Based Proposals](#slice-based-proposals)
- [Transport-Based Proposals](#transport-based-proposals)
- [Gradient and MALA Proposals](#gradient-and-mala-proposals)
- [Large-Jump and Mode-Jump Proposals](#large-jump-and-mode-jump-proposals)
- [Preconditioned Crank-Nicolson Proposals](#preconditioned-crank-nicolson-proposals)
- [Irreversible Guided Random Walks](#irreversible-guided-random-walks)
- [Delayed-Acceptance and ML-Assisted Runs](#delayed-acceptance-and-ml-assisted-runs)
- [Adaptive Operator Weighting and Scheduling](#adaptive-operator-weighting-and-scheduling)

## Adapters

Many custom operators are built around an adapter layer, which exposes parts of the BEAST state as mutable and immutable vectors, often after a transform, so generic proposal machinery can work on scalar parameters, simplex parameters, tree heights, local tree geometry, or MAP/cube summaries.

Relevant classes:

- [`adapters.TreeHeightAdapter`](src/main/java/adapters/TreeHeightAdapter.java)
- [`adapters.MutableTreeHeightAdapter`](src/main/java/adapters/MutableTreeHeightAdapter.java)
- [`adapters.CubeAdapter`](src/main/java/adapters/CubeAdapter.java)
- [`adapters.TaxaDistanceAdapterGenerator`](src/main/java/adapters/TaxaDistanceAdapterGenerator.java)
- [`adapters.TreeTripletAdapter`](src/main/java/adapters/TreeTripletAdapter.java)
- [`adapters.LocalTreeAdapter`](src/main/java/adapters/LocalTreeAdapter.java)
- [`adapters.NodePositionAdapter`](src/main/java/adapters/NodePositionAdapter.java)

Relevant transform classes:

- [`transforms.RealScalarLogTransform`](src/main/java/transforms/RealScalarLogTransform.java)
- [`transforms.RealScalarSigmoidTransform`](src/main/java/transforms/RealScalarSigmoidTransform.java)
- [`transforms.RealVectorLogTransform`](src/main/java/transforms/RealVectorLogTransform.java)
- [`transforms.SimplexTransform`](src/main/java/transforms/SimplexTransform.java)
- [`transforms.IntVectorIdentityTransform`](src/main/java/transforms/IntVectorIdentityTransform.java)

## Learned Conditional Proposals

The adaptive operator learns a conditional proposal distribution from observed mutable and immutable adapter vectors, then, after burn-in and training, samples new mutable values conditional on the current immutable state.

Relevant classes:

- [`adaptiveoperators.AdaptiveOperator`](src/main/java/adaptiveoperators/AdaptiveOperator.java)
- [`adaptiveoperators.MultivariateNormalSampler`](src/main/java/adaptiveoperators/MultivariateNormalSampler.java)
- [`adaptiveoperators.GaussianMixtureSampler`](src/main/java/adaptiveoperators/GaussianMixtureSampler.java)
- [`adaptiveoperators.NeuralGaussianMixtureSampler`](src/main/java/adaptiveoperators/NeuralGaussianMixtureSampler.java)

## Learned Tree-Distance Proposals

These operators learn distributions over taxon-pair or taxon-triplet distances, then propose tree edits by changing those distances directly.

Relevant classes:

- [`adaptiveoperators.TaxaDistanceOperator`](src/main/java/adaptiveoperators/TaxaDistanceOperator.java)
- [`adaptiveoperators.TripletDistanceOperator`](src/main/java/adaptiveoperators/TripletDistanceOperator.java)
- [`adaptiveoperators.LogNormalModel`](src/main/java/adaptiveoperators/LogNormalModel.java)
- [`adaptiveoperators.NeuralLogNormalModel`](src/main/java/adaptiveoperators/NeuralLogNormalModel.java)

## Slice-Based Proposals

The slice operators all expose adapted BEAST state as a vector, choose a one-dimensional direction through that vector space, draw a slice level from the current posterior, then use step-out and shrinkage along that line until they find an acceptable point.

There are three variants:

- `StepOutShrinkSliceOperator` is coordinate-wise: it chooses one mutable adapter coordinate, maintains an adaptive window size for that coordinate, and updates only that value.
- `MultivariateStepOutShrinkSliceOperator` is random-direction slice sampling: it draws a normalized Gaussian direction across all mutable adapter coordinates, applies per-coordinate learning-rate scaling, and slices along that line.
- `LinCombSliceOperator` is empirical-direction slice sampling: after burn-in it stores recent adapted states, chooses two previous states, and slices along their difference vector. This turns recent posterior movement into proposal directions.

Relevant classes:

- [`slice.StepOutShrinkSliceOperator`](src/main/java/slice/StepOutShrinkSliceOperator.java)
- [`slice.MultivariateStepOutShrinkSliceOperator`](src/main/java/slice/MultivariateStepOutShrinkSliceOperator.java)
- [`slice.LinCombSliceOperator`](src/main/java/slice/LinCombSliceOperator.java)
- [`mcmc.SliceMCMC`](src/main/java/mcmc/SliceMCMC.java)

## Transport-Based Proposals

Tree-topology cube parameterizations can be multi-modal, which makes local moves inefficient. The transport operators look at two reference taxa and a subtree on the path between the references. It then approximates the posterior slice for all the possible (continuous) attachment points of the subtree and builds a triangular (Knothe-Rosenblatt) transport map that sends this approximation to a standard Gaussian. A move perturbs the Gaussian-space point, transports it back to distance space, and computes an exact log Hastings-ratio correction from the map's Jacobian. See [`transport/approach.md`](src/main/java/transport/approach.md) for details.

- `NodeTransportOperator` reattaches a subtree by resampling its attachment point along the path between two reference leaves.
- `GuidedNodeTransportOperator` picks reference leaves and a subtree node via sequence-distance-weighted triplet selection, then resamples its attachment point in the same way.

Relevant classes:

- [`transport.NodeTransportOperator`](src/main/java/transport/NodeTransportOperator.java)
- [`transport.GuidedNodeTransportOperator`](src/main/java/transport/GuidedNodeTransportOperator.java)
- [`transport.PartialCubeTransportOperator`](src/main/java/transport/PartialCubeTransportOperator.java)
- [`transport.UnivariateOptimalTransportMap`](src/main/java/transport/UnivariateOptimalTransportMap.java)

## Gradient and MALA Proposals

The MALA family proposes adapter vectors using approximate-gradient-guided proposals. `MALAOperator` learns a neural-network or Gaussian approximation to the gradient during a training phase, then uses a learned covariance for preconditioned proposals. `FisherMALAOperator` adds Fisher-style preconditioning and adaptive step-size behavior, and the MAP-guided variants use MAP/cube summaries to guide moves.

Relevant classes:

- [`mala.MALAOperator`](src/main/java/mala/MALAOperator.java)
- [`mala.FisherMALAOperator`](src/main/java/mala/FisherMALAOperator.java)
- [`mala.MAPGuidedMALAOperator`](src/main/java/mala/MAPGuidedMALAOperator.java)
- [`mala.CubeMAPGuidedMALAOperator`](src/main/java/mala/CubeMAPGuidedMALAOperator.java)
- [`mcmc.MalaMCMC`](src/main/java/mcmc/MalaMCMC.java)

## Large-Jump and Mode-Jump Proposals

Large-jump operators separate jump coordinates from optimization coordinates: a proposal first makes a large move in one adapter group, performs local random-walk optimization in another group, adds covariance-scaled noise, and computes the reverse construction for the Hastings correction. The unified variant also supports MAP jump candidates and transport noise controls.

Relevant classes:

- [`largejump.LargeJumpMALAOperator`](src/main/java/largejump/LargeJumpMALAOperator.java)
- [`largejump.UnifiedLargeJumpMALAOperator`](src/main/java/largejump/UnifiedLargeJumpMALAOperator.java)
- [`mcmc.JumpMCMC`](src/main/java/mcmc/JumpMCMC.java)

## Irreversible Guided Random Walks

The irreversible guided random-walk operator chooses an adapted coordinate, moves in a persistent direction, and flips that direction on rejection.

Relevant class:

- [`irreversible.GuidedRandomWalkOperator`](src/main/java/irreversible/GuidedRandomWalkOperator.java)

## Preconditioned Crank-Nicolson Proposals

The pCN operators learn a centered covariance over adapted mutable values and then propose by shrinking the current state toward the learned mean plus a covariance-scaled perturbation. The guided mixed variant combines this with guided adapter information.

Relevant classes:

- [`adaptiveoperators.PreconditionedCrankNicolsonOperator`](src/main/java/adaptiveoperators/PreconditionedCrankNicolsonOperator.java)


## Delayed-Acceptance and ML-Assisted Runs

The delayed-acceptance classes use an approximate posterior, including an MLP-backed approximation, to stage accept/reject work.

Relevant classes:

- [`delayedacceptance.DelayedAcceptanceMCMC`](src/main/java/delayedacceptance/DelayedAcceptanceMCMC.java)
- [`delayedacceptance.MLPPosteriorApproximation`](src/main/java/delayedacceptance/MLPPosteriorApproximation.java)

## Adaptive Operator Weighting and Scheduling

Two adaptive-control pieces are implemented separately from proposal kernels. `AdaptiveWeightOperator` chooses among child operators and learns their weights through a `WeightScheme`. `DualAveragingOperatorSchedule` provides better schedule-level adaptation compared to Robinson-Monro.

Relevant classes:

- [`schedule.DualAveragingOperatorSchedule`](src/main/java/schedule/DualAveragingOperatorSchedule.java)
- [`weightoptimization.AdaptiveWeightOperator`](src/main/java/weightoptimization/AdaptiveWeightOperator.java)
- [`weightoptimization.RunningAverageScheme`](src/main/java/weightoptimization/RunningAverageScheme.java)
