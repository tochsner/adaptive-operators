package adaptiveoperators;

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
 * Models all taxon pair distances jointly with a single MLP. The network takes the
 * (log-)distances of every pair as input, with the chosen pair's entry masked to zero,
 * and produces a mean and log-variance of the log-distance for each pair. Only the
 * chosen pair's outputs are used to parameterize a shifted log-normal and only the
 * chosen pair's negative log-likelihood contributes to the training loss. The network is
 * trained incrementally as observations are recorded.
 */
public class NeuralLogNormalModel implements TaxaDistanceModel, AutoCloseable {

    private static final int BATCH_SIZE = 64;
    private static final int FRESH_BATCH_SIZE = BATCH_SIZE / 4;
    private static final int REPLAY_BATCH_SIZE = BATCH_SIZE - FRESH_BATCH_SIZE;
    private static final int REPLAY_CAPACITY = 10_000;
    private static final int TRAINING_PASSES = 1;
    private static final float LEARNING_RATE = 0.001f;
    private static final float WEIGHT_DECAY = 0.0001f;
    private static final double LOG_VARIANCE_MIN = -5.0;
    private static final double LOG_VARIANCE_MAX = 5.0;
    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);

    private final int numPairs;
    private final int inputDim;
    private final int outputDim;
    private final double[] offset;

    private final NDManager manager;
    private final Model model;
    private final Trainer trainer;

    // pending fresh examples awaiting a training step
    private final float[] inputBatch;
    private final float[] targetBatch;
    private final int[] pairBatch;
    private int batchSize;

    // replay buffer of past examples
    private final float[] replayInputs;
    private final float[] replayTargets;
    private final int[] replayPairs;
    private int replaySize;
    private int replayNextIndex;

    private final Random random = new Random();

    // cache of the last prediction, valid until the next training step
    private long trainingVersion;
    private long cachedVersion = -1;
    private float[] cachedInput;
    private double cachedMean;
    private double cachedVariance;

    private boolean closed;

    public NeuralLogNormalModel(double[] offsets) {
        for (double offset : offsets) {
            if (!Double.isFinite(offset) || offset < 0.0) {
                throw new IllegalArgumentException("Log-normal offset must be a finite non-negative value");
            }
        }

        this.numPairs = offsets.length;
        this.inputDim = offsets.length;
        this.outputDim = 2 * offsets.length;
        this.offset = offsets.clone();

        this.manager = NDManager.newBaseManager("PyTorch");
        this.model = Model.newInstance("neural-log-normal", "PyTorch");
        this.model.setBlock(createBlock(inputDim, outputDim));
        this.trainer = model.newTrainer(setupTrainingConfig());
        this.trainer.initialize(new Shape(1, inputDim));

        this.inputBatch = new float[FRESH_BATCH_SIZE * inputDim];
        this.targetBatch = new float[FRESH_BATCH_SIZE];
        this.pairBatch = new int[FRESH_BATCH_SIZE];
        this.replayInputs = new float[REPLAY_CAPACITY * inputDim];
        this.replayTargets = new float[REPLAY_CAPACITY];
        this.replayPairs = new int[REPLAY_CAPACITY];
    }

    @Override
    public void record(double[] distances, int pairIndex) {
        ensureOpen();

        double shiftedDistance = distances[pairIndex] - this.offset[pairIndex];
        if (!Double.isFinite(shiftedDistance) || shiftedDistance <= 0.0) {
            return;
        }

        float[] input = buildInput(distances, pairIndex);
        if (input == null) return;

        System.arraycopy(input, 0, this.inputBatch, batchSize * inputDim, inputDim);
        this.targetBatch[batchSize] = (float) Math.log(shiftedDistance);
        this.pairBatch[batchSize] = pairIndex;
        batchSize++;

        if (batchSize == FRESH_BATCH_SIZE) {
            trainBatch();
        }
    }

    @Override
    public double sample(double[] distances, int pairIndex, Random random, double scaleFactor) {
        predict(distances, pairIndex);
        double standardDeviation = Math.sqrt(scaleFactor * cachedVariance);
        return this.offset[pairIndex] + Math.exp(cachedMean + standardDeviation * random.nextGaussian());
    }

    @Override
    public double logDensity(double[] distances, int pairIndex, double value, double scaleFactor) {
        double shiftedDistance = value - this.offset[pairIndex];
        if (!Double.isFinite(shiftedDistance) || shiftedDistance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        predict(distances, pairIndex);
        double variance = scaleFactor * cachedVariance;
        double logDistance = Math.log(shiftedDistance);
        double diff = logDistance - cachedMean;
        return -Math.log(shiftedDistance)
                - 0.5 * (Math.log(2.0 * Math.PI * variance) + diff * diff / variance);
    }

    private void predict(double[] distances, int pairIndex) {
        ensureOpen();
        float[] input = buildInput(distances, pairIndex);

        if (cachedVersion == trainingVersion && Arrays.equals(input, cachedInput)) {
            return;
        }

        float[] output;
        try (NDManager inferenceManager = manager.newSubManager()) {
            NDArray in = inferenceManager.create(input, new Shape(1, inputDim));
            NDArray out = trainer.evaluate(new NDList(in)).singletonOrThrow();
            output = out.toFloatArray();
        }

        double logVariance = output[numPairs + pairIndex];
        logVariance = Math.max(LOG_VARIANCE_MIN, Math.min(LOG_VARIANCE_MAX, logVariance));

        cachedVersion = trainingVersion;
        cachedInput = input;
        cachedMean = output[pairIndex];
        cachedVariance = Math.exp(logVariance);
    }

    /**
     * Builds the network input from the current distances, masking the chosen pair to
     * zero and feeding the log of the shifted distance for the remaining pairs. Returns
     * {@code null} if any non-chosen distance is non-finite.
     */
    private float[] buildInput(double[] distances, int pairIndex) {
        float[] input = new float[inputDim];
        for (int i = 0; i < numPairs; i++) {
            if (i == pairIndex) {
                input[i] = 0.0f;
                continue;
            }
            double shifted = distances[i] - this.offset[i];
            input[i] = (Double.isFinite(shifted) && shifted > 0.0) ? (float) Math.log(shifted) : 0.0f;
        }
        return input;
    }

    private void trainBatch() {
        ensureOpen();
        int sampledReplaySize = Math.min(REPLAY_BATCH_SIZE, replaySize);
        int trainingSize = batchSize + sampledReplaySize;

        float[] inputs = new float[trainingSize * inputDim];
        // labels hold the target log-distance in the first half and the pair mask in the second
        float[] labels = new float[trainingSize * outputDim];
        System.arraycopy(inputBatch, 0, inputs, 0, batchSize * inputDim);
        for (int i = 0; i < batchSize; i++) {
            labels[i * outputDim + pairBatch[i]] = targetBatch[i];
            labels[i * outputDim + numPairs + pairBatch[i]] = 1.0f;
        }
        copyReplaySamples(inputs, labels, sampledReplaySize);

        try (NDManager batchManager = manager.newSubManager()) {
            NDArray input = batchManager.create(inputs, new Shape(trainingSize, inputDim));
            NDArray label = batchManager.create(labels, new Shape(trainingSize, outputDim));
            for (int i = 0; i < TRAINING_PASSES; i++) {
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    NDArray predictions = trainer.forward(new NDList(input)).singletonOrThrow();
                    NDArray loss = trainer.getLoss().evaluate(new NDList(label), new NDList(predictions));
                    collector.backward(loss);
                }
                trainer.step();
            }
        }

        appendFreshBatchToReplayBuffer();
        batchSize = 0;
        trainingVersion++;
    }

    private void copyReplaySamples(float[] inputs, float[] labels, int sampledReplaySize) {
        int inputOffset = batchSize * inputDim;
        int labelOffset = batchSize * outputDim;
        for (int i = 0; i < sampledReplaySize; i++) {
            int replayIndex = random.nextInt(replaySize);
            System.arraycopy(
                    replayInputs, replayIndex * inputDim,
                    inputs, inputOffset + i * inputDim, inputDim);
            int pair = replayPairs[replayIndex];
            labels[labelOffset + i * outputDim + pair] = replayTargets[replayIndex];
            labels[labelOffset + i * outputDim + numPairs + pair] = 1.0f;
        }
    }

    private void appendFreshBatchToReplayBuffer() {
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(
                    inputBatch, i * inputDim,
                    replayInputs, replayNextIndex * inputDim, inputDim);
            replayTargets[replayNextIndex] = targetBatch[i];
            replayPairs[replayNextIndex] = pairBatch[i];
            replayNextIndex = (replayNextIndex + 1) % REPLAY_CAPACITY;
            replaySize = Math.min(replaySize + 1, REPLAY_CAPACITY);
        }
    }

    private static SequentialBlock createBlock(int inputDim, int outputDim) {
        return new SequentialBlock()
                .add(Linear.builder().setUnits(inputDim + outputDim).build())
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
        return new DefaultTrainingConfig(new MaskedGaussianLoss())
                .optOptimizer(optimizer);
    }

    /**
     * Negative log-likelihood of the chosen pair's log-distance under the Gaussian
     * predicted for that pair. The label's second half masks the loss to the chosen pair.
     */
    private final class MaskedGaussianLoss extends Loss {

        private MaskedGaussianLoss() {
            super("MaskedGaussianLoss");
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray label = labels.singletonOrThrow();
            NDArray params = predictions.singletonOrThrow();

            NDArray targets = label.get(new NDIndex(":, 0:" + numPairs));
            NDArray mask = label.get(new NDIndex(":, " + numPairs + ":"));
            NDArray means = params.get(new NDIndex(":, 0:" + numPairs));
            NDArray logVariances = params.get(new NDIndex(":, " + numPairs + ":"))
                    .clip(LOG_VARIANCE_MIN, LOG_VARIANCE_MAX);

            NDArray diff = targets.sub(means);
            NDArray perPairNll = logVariances
                    .add(diff.square().mul(logVariances.neg().exp()))
                    .add(LOG_TWO_PI)
                    .mul(0.5);

            // keep only the chosen pair's contribution; one pair is masked per row
            return perPairNll.mul(mask).sum(new int[]{1}).mean();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("NeuralLogNormalModel is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            trainer.close();
            model.close();
            manager.close();
        }
    }
}
