package ml;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
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
 * Trains an MLP in an incremental fashion.
 */
public class MLP implements AutoCloseable {

    private static final int BATCH_SIZE = 32;
    private static final int FRESH_BATCH_SIZE = BATCH_SIZE / 2;
    private static final int REPLAY_BATCH_SIZE = BATCH_SIZE - FRESH_BATCH_SIZE;
    private static final int REPLAY_CAPACITY = 10_000;
    private static final int TRAINING_PASSES = 2;
    private static final float LEARNING_RATE = 0.01f;
    private final int inputDim;
    private final int outputDim;
    private final NDManager manager;
    private final Model model;
    private final Trainer trainer;
    private final float[] inputBatch;
    private final float[] outputBatch;
    private final Random random;
    private final float[] replayInputs;
    private final float[] replayOutputs;
    private int batchSize;
    private int replaySize;
    private int replayNextIndex;
    private boolean closed;

    /**
     * Creates an MLP with the given input and output dimensions and initializes its
     * trainer. The network is trained incrementally as samples are recorded.
     */
    public MLP(int inputDim, int outputDim) {
        if (inputDim <= 0) {
            throw new IllegalArgumentException("inputDim must be positive");
        }
        if (outputDim <= 0) {
            throw new IllegalArgumentException("outputDim must be positive");
        }
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.manager = NDManager.newBaseManager("PyTorch");
        this.model = Model.newInstance("mlp", "PyTorch");
        this.model.setBlock(createBlock(inputDim, outputDim));
        this.trainer = model.newTrainer(setupTrainingConfig());
        this.trainer.initialize(new Shape(1, inputDim));
        this.inputBatch = new float[FRESH_BATCH_SIZE * inputDim];
        this.outputBatch = new float[FRESH_BATCH_SIZE * outputDim];
        this.random = new Random();
        this.replayInputs = new float[REPLAY_CAPACITY * inputDim];
        this.replayOutputs = new float[REPLAY_CAPACITY * outputDim];
    }

    /**
     * Buffers a single training example and triggers a training step once a full batch
     * has accumulated. Examples containing non-finite values are silently ignored.
     */
    public void record(double[] inputs, double[] output) {
        if (!Arrays.stream(inputs).allMatch(Double::isFinite)) return;
        if (!Arrays.stream(output).allMatch(Double::isFinite)) return;

        ensureOpen();
        validateInput(inputs);
        validateOutput(output);

        copy(inputs, inputBatch, batchSize * inputDim);
        copy(output, outputBatch, batchSize * outputDim);
        batchSize++;

        if (batchSize == FRESH_BATCH_SIZE) {
            trainBatch();
        }
    }

    /**
     * Runs a forward pass and returns the network's prediction for the given input.
     */
    public double[] runInference(double[] inputs) {
        ensureOpen();
        validateInput(inputs);

        try (NDManager inferenceManager = manager.newSubManager()) {
            NDArray input = inferenceManager.create(toFloatArray(inputs), new Shape(1, inputDim));
            NDArray output = trainer.evaluate(new NDList(input)).singletonOrThrow();
            return toDoubleArray(output.toFloatArray());
        }
    }

    /**
     * Computes the gradient of the scalar network output with respect to the input.
     * Only supported when {@code outputDim == 1}.
     */
    public double[] getGradient(double[] inputs) {
        ensureOpen();
        validateInput(inputs);

        if (this.outputDim != 1) {
            throw new UnsupportedOperationException("Gradients are only supported for univariate functions.");
        }

        try (NDManager gradientManager = manager.newSubManager();
                GradientCollector collector = trainer.newGradientCollector()) {
            NDArray input = gradientManager.create(toFloatArray(inputs), new Shape(1, inputDim));
            input.setRequiresGradient(true);
            NDArray output = trainer.forward(new NDList(input)).singletonOrThrow();
            collector.backward(output);
            return toDoubleArray(input.getGradient().toFloatArray());
        }
    }

    private void trainBatch() {
        ensureOpen();
        int sampledReplaySize = Math.min(REPLAY_BATCH_SIZE, replaySize);
        int trainingSize = batchSize + sampledReplaySize;
        float[] inputs = new float[trainingSize * inputDim];
        float[] outputs = new float[trainingSize * outputDim];
        System.arraycopy(inputBatch, 0, inputs, 0, batchSize * inputDim);
        System.arraycopy(outputBatch, 0, outputs, 0, batchSize * outputDim);
        copyReplaySamples(inputs, outputs, sampledReplaySize);

        try (NDManager batchManager = manager.newSubManager()) {
            NDArray input = batchManager.create(inputs, new Shape(trainingSize, inputDim));
            NDArray labels = batchManager.create(outputs, new Shape(trainingSize, outputDim));
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
        }
    }

    private void copyReplaySamples(float[] inputs, float[] outputs, int sampledReplaySize) {
        int inputTargetOffset = batchSize * inputDim;
        int outputTargetOffset = batchSize * outputDim;
        for (int i = 0; i < sampledReplaySize; i++) {
            int replayIndex = random.nextInt(replaySize);
            System.arraycopy(
                    replayInputs,
                    replayIndex * inputDim,
                    inputs,
                    inputTargetOffset + i * inputDim,
                    inputDim);
            System.arraycopy(
                    replayOutputs,
                    replayIndex * outputDim,
                    outputs,
                    outputTargetOffset + i * outputDim,
                    outputDim);
        }
    }

    private void appendFreshBatchToReplayBuffer() {
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(
                    inputBatch,
                    i * inputDim,
                    replayInputs,
                    replayNextIndex * inputDim,
                    inputDim);
            System.arraycopy(
                    outputBatch,
                    i * outputDim,
                    replayOutputs,
                    replayNextIndex * outputDim,
                    outputDim);
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

    private static DefaultTrainingConfig setupTrainingConfig() {
        Adam optimizer =
                Adam.builder()
                        .optLearningRateTracker(Tracker.fixed(LEARNING_RATE))
                        .optWeightDecays(0.01f)
                        .optClipGrad(1.0f)
                        .build();
        return new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(optimizer);
    }

    private void validateInput(double[] inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs must not be null");
        }
        if (inputs.length != inputDim) {
            throw new IllegalArgumentException(
                    "inputs must have length " + inputDim + ", got " + inputs.length);
        }
    }

    private void validateOutput(double[] output) {
        if (output == null) {
            throw new IllegalArgumentException("output must not be null");
        }
        if (output.length != outputDim) {
            throw new IllegalArgumentException(
                    "output must have length " + outputDim + ", got " + output.length);
        }
    }

    private static void copy(double[] source, float[] target, int offset) {
        for (int i = 0; i < source.length; i++) {
            target[offset + i] = (float) source[i];
        }
    }

    private static float[] toFloatArray(double[] values) {
        float[] result = new float[values.length];
        copy(values, result, 0);
        return result;
    }

    private static double[] toDoubleArray(float[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("MLP is closed");
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
}
