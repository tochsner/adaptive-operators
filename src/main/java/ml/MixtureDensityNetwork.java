package ml;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;

import java.util.Arrays;
import java.util.Random;

/**
 * Trains a mixture density network in an incremental fashion. Given a vector of
 * conditions, the network predicts the parameters of a diagonal-covariance Gaussian
 * mixture over the values: component logits, component means, and component
 * log-standard-deviations. The network is optimized by minimizing the negative
 * log-likelihood of the recorded (conditions, values) pairs.
 */
public class MixtureDensityNetwork implements AutoCloseable {

    private static final int BATCH_SIZE = 128;
    private static final int FRESH_BATCH_SIZE = BATCH_SIZE / 4;
    private static final int REPLAY_BATCH_SIZE = BATCH_SIZE - FRESH_BATCH_SIZE;
    private static final int REPLAY_CAPACITY = 10_000;
    private static final int TRAINING_PASSES = 1;
    private static final float LEARNING_RATE = 0.001f;
    private static final float WEIGHT_DECAY = 0.0001f;
    private static final double LOG_SIGMA_MIN = -2.0;
    private static final double LOG_SIGMA_MAX = 2.0;
    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);

    private final int numConditions;
    private final int numValues;
    private final int numComponents;
    private final int featureDim;
    private final int paramDim;
    private final NDManager manager;
    private final Model model;
    private final Trainer trainer;
    private final float[] featureBatch;
    private final float[] valueBatch;
    private final Random random;
    private final float[] replayFeatures;
    private final float[] replayValues;
    private int batchSize;
    private int replaySize;
    private int replayNextIndex;
    private boolean closed;

    // cache of the last prediction, valid until the next training step
    private long trainingVersion;
    private long cachedVersion = -1;
    private float[] cachedFeatures;
    private MixtureParameters cachedParameters;

    /**
     * Creates a mixture density network and initializes its trainer. The network is
     * trained incrementally as samples are recorded. A network without conditions is
     * supported and is fed a constant input instead.
     */
    public MixtureDensityNetwork(int numConditions, int numValues, int numComponents) {
        if (numConditions < 0) {
            throw new IllegalArgumentException("numConditions must be non-negative");
        }
        if (numValues <= 0) {
            throw new IllegalArgumentException("numValues must be positive");
        }
        if (numComponents <= 0) {
            throw new IllegalArgumentException("numComponents must be positive");
        }
        this.numConditions = numConditions;
        this.numValues = numValues;
        this.numComponents = numComponents;
        this.featureDim = Math.max(numConditions, 1);
        this.paramDim = numComponents * (1 + 2 * numValues);
        this.manager = NDManager.newBaseManager("PyTorch");
        this.model = Model.newInstance("mixture-density-network", "PyTorch");
        this.model.setBlock(createBlock(featureDim, paramDim));
        this.trainer = model.newTrainer(setupTrainingConfig());
        this.trainer.initialize(new Shape(1, featureDim));
        this.featureBatch = new float[FRESH_BATCH_SIZE * featureDim];
        this.valueBatch = new float[FRESH_BATCH_SIZE * numValues];
        this.random = new Random();
        this.replayFeatures = new float[REPLAY_CAPACITY * featureDim];
        this.replayValues = new float[REPLAY_CAPACITY * numValues];
    }

    /**
     * Buffers a single training example and triggers a training step once a full batch
     * has accumulated. Examples containing non-finite values are silently ignored.
     */
    public void record(double[] conditions, double[] values) {
        if (!Arrays.stream(conditions).allMatch(Double::isFinite)) return;
        if (!Arrays.stream(values).allMatch(Double::isFinite)) return;

        ensureOpen();
        validateConditions(conditions);
        validateValues(values);

        copyFeatures(conditions, featureBatch, batchSize * featureDim);
        copy(values, valueBatch, batchSize * numValues);
        batchSize++;

        if (batchSize == FRESH_BATCH_SIZE) {
            trainBatch();
        }
    }

    /**
     * Runs a forward pass and returns the predicted mixture parameters for the given
     * conditions.
     */
    public MixtureParameters predict(double[] conditions) {
        ensureOpen();
        validateConditions(conditions);

        float[] features = new float[featureDim];
        copyFeatures(conditions, features, 0);

        if (cachedVersion == trainingVersion && Arrays.equals(features, cachedFeatures)) {
            return cachedParameters;
        }

        float[] params;
        try (NDManager inferenceManager = manager.newSubManager()) {
            NDArray input = inferenceManager.create(features, new Shape(1, featureDim));
            NDArray output = trainer.evaluate(new NDList(input)).singletonOrThrow();
            params = output.toFloatArray();
        }

        cachedVersion = trainingVersion;
        cachedFeatures = features;
        cachedParameters = parseParameters(params);
        return cachedParameters;
    }

    private MixtureParameters parseParameters(float[] params) {
        double[] weights = new double[numComponents];
        double[][] means = new double[numComponents][numValues];
        double[][] sigmas = new double[numComponents][numValues];

        // weights are the softmax of the logits, computed stably via the max logit
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < numComponents; k++) {
            maxLogit = Math.max(maxLogit, params[k]);
        }
        double weightSum = 0.0;
        for (int k = 0; k < numComponents; k++) {
            weights[k] = Math.exp(params[k] - maxLogit);
            weightSum += weights[k];
        }
        for (int k = 0; k < numComponents; k++) {
            weights[k] /= weightSum;
        }

        int meanOffset = numComponents;
        int logSigmaOffset = numComponents + numComponents * numValues;
        for (int k = 0; k < numComponents; k++) {
            for (int i = 0; i < numValues; i++) {
                means[k][i] = params[meanOffset + k * numValues + i];
                double logSigma = params[logSigmaOffset + k * numValues + i];
                logSigma = Math.max(LOG_SIGMA_MIN, Math.min(LOG_SIGMA_MAX, logSigma));
                sigmas[k][i] = Math.exp(logSigma);
            }
        }

        return new MixtureParameters(weights, means, sigmas);
    }

    private void trainBatch() {
        ensureOpen();
        int sampledReplaySize = Math.min(REPLAY_BATCH_SIZE, replaySize);
        int trainingSize = batchSize + sampledReplaySize;
        float[] features = new float[trainingSize * featureDim];
        float[] values = new float[trainingSize * numValues];
        System.arraycopy(featureBatch, 0, features, 0, batchSize * featureDim);
        System.arraycopy(valueBatch, 0, values, 0, batchSize * numValues);
        copyReplaySamples(features, values, sampledReplaySize);

        try (NDManager batchManager = manager.newSubManager()) {
            NDArray input = batchManager.create(features, new Shape(trainingSize, featureDim));
            NDArray labels = batchManager.create(values, new Shape(trainingSize, numValues));
            for (int i = 0; i < TRAINING_PASSES; i++) {
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    NDArray predictions = trainer.forward(new NDList(input)).singletonOrThrow();
                    NDArray loss = trainer.getLoss().evaluate(new NDList(labels), new NDList(predictions));
                    collector.backward(loss);
                }
                trainer.step();
            }
            appendFreshBatchToReplayBuffer();
            batchSize = 0;
            trainingVersion++;
        }
    }

    private void copyReplaySamples(float[] features, float[] values, int sampledReplaySize) {
        int featureTargetOffset = batchSize * featureDim;
        int valueTargetOffset = batchSize * numValues;
        for (int i = 0; i < sampledReplaySize; i++) {
            int replayIndex = random.nextInt(replaySize);
            System.arraycopy(
                    replayFeatures,
                    replayIndex * featureDim,
                    features,
                    featureTargetOffset + i * featureDim,
                    featureDim);
            System.arraycopy(
                    replayValues,
                    replayIndex * numValues,
                    values,
                    valueTargetOffset + i * numValues,
                    numValues);
        }
    }

    private void appendFreshBatchToReplayBuffer() {
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(
                    featureBatch,
                    i * featureDim,
                    replayFeatures,
                    replayNextIndex * featureDim,
                    featureDim);
            System.arraycopy(
                    valueBatch,
                    i * numValues,
                    replayValues,
                    replayNextIndex * numValues,
                    numValues);
            replayNextIndex = (replayNextIndex + 1) % REPLAY_CAPACITY;
            replaySize = Math.min(replaySize + 1, REPLAY_CAPACITY);
        }
    }

    private static SequentialBlock createBlock(int inputDim, int outputDim) {
        return new SequentialBlock()
                .add(Linear.builder().setUnits(128).build())
                .add(Activation.tanhBlock())
                .add(Linear.builder().setUnits(outputDim).build());
    }

    private DefaultTrainingConfig setupTrainingConfig() {
        Adam optimizer =
                Adam.builder()
                        .optLearningRateTracker(Tracker.fixed(LEARNING_RATE))
                        .optWeightDecays(WEIGHT_DECAY)
                        .optClipGrad(1.0f)
                        .build();
        return new DefaultTrainingConfig(new MixtureDensityLoss())
                .optOptimizer(optimizer);
    }

    /**
     * Negative log-likelihood of the labels under the Gaussian mixture parameterized by
     * the network output.
     */
    private final class MixtureDensityLoss extends Loss {

        private MixtureDensityLoss() {
            super("MixtureDensityLoss");
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray values = labels.singletonOrThrow();
            NDArray params = predictions.singletonOrThrow();
            long batch = params.getShape().get(0);

            int meanOffset = numComponents;
            int logSigmaOffset = numComponents + numComponents * numValues;

            NDArray logWeights = params.get(new NDIndex(":, 0:" + numComponents)).logSoftmax(1);
            NDArray means = params.get(new NDIndex(":, " + meanOffset + ":" + logSigmaOffset))
                    .reshape(batch, numComponents, numValues);
            NDArray logSigmas = params.get(new NDIndex(":, " + logSigmaOffset + ":"))
                    .reshape(batch, numComponents, numValues)
                    .clip(LOG_SIGMA_MIN, LOG_SIGMA_MAX);

            // per-component diagonal Gaussian log-density, summed over the value dimensions
            NDArray standardized = values.reshape(batch, 1, numValues).sub(means).div(logSigmas.exp());
            NDArray componentLogDensities = logSigmas
                    .add(standardized.square().mul(0.5))
                    .add(0.5 * LOG_TWO_PI)
                    .sum(new int[]{2})
                    .neg();

            // mixture log-likelihood via a stable log-sum-exp over the components
            NDArray joint = logWeights.add(componentLogDensities);
            NDArray max = joint.max(new int[]{1}, true);
            NDArray logLikelihood = joint.sub(max).exp().sum(new int[]{1}, true).log().add(max);

            return logLikelihood.neg().mean();
        }
    }

    private void validateConditions(double[] conditions) {
        if (conditions == null) {
            throw new IllegalArgumentException("conditions must not be null");
        }
        if (conditions.length != numConditions) {
            throw new IllegalArgumentException(
                    "conditions must have length " + numConditions + ", got " + conditions.length);
        }
    }

    private void validateValues(double[] values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        if (values.length != numValues) {
            throw new IllegalArgumentException(
                    "values must have length " + numValues + ", got " + values.length);
        }
    }

    private void copyFeatures(double[] conditions, float[] target, int offset) {
        if (numConditions == 0) {
            // feed a constant input so that the network reduces to an unconditional mixture
            target[offset] = 0.0f;
            return;
        }
        copy(conditions, target, offset);
    }

    private static void copy(double[] source, float[] target, int offset) {
        for (int i = 0; i < source.length; i++) {
            target[offset + i] = (float) source[i];
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("MixtureDensityNetwork is closed");
        }
    }

    /**
     * Releases the underlying trainer, model, and native memory manager. The instance
     * must not be used after being closed.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            trainer.close();
            model.close();
            manager.close();
        }
    }

    /**
     * Parameters of a diagonal-covariance Gaussian mixture: normalized component
     * weights, per-component means, and per-component standard deviations.
     */
    public static final class MixtureParameters {

        public final double[] weights;
        public final double[][] means;
        public final double[][] sigmas;

        private MixtureParameters(double[] weights, double[][] means, double[][] sigmas) {
            this.weights = weights;
            this.means = means;
            this.sigmas = sigmas;
        }
    }
}
