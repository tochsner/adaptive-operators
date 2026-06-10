package ml;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.Arrays;

/**
 * Trains an MLP in an incremental fashion.
 */
public class MLP implements AutoCloseable {

    private static final int BATCH_SIZE = 128;
    private static final float LEARNING_RATE = 0.01f;
    private final int inputDim;
    private final int outputDim;
    private final NDManager manager;
    private final Model model;
    private final Trainer trainer;
    private final float[] inputBatch;
    private final float[] outputBatch;
    private int batchSize;
    private boolean closed;

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
        this.model.setBlock(createBlock(outputDim));
        this.trainer = model.newTrainer(setupTrainingConfig());
        this.trainer.setMetrics(new Metrics());
        this.trainer.initialize(new Shape(1, inputDim));
        this.inputBatch = new float[BATCH_SIZE * inputDim];
        this.outputBatch = new float[BATCH_SIZE * outputDim];
    }

    public void record(double[] inputs, double[] output) {
        ensureOpen();
        validateInput(inputs);
        validateOutput(output);

        copy(inputs, inputBatch, batchSize * inputDim);
        copy(output, outputBatch, batchSize * outputDim);
        batchSize++;

        if (batchSize == BATCH_SIZE) {
            trainBatch();
        }
    }

    public double[] runInference(double[] inputs) {
        ensureOpen();
        validateInput(inputs);

        try (NDManager inferenceManager = manager.newSubManager()) {
            NDArray input = inferenceManager.create(toFloatArray(inputs), new Shape(1, inputDim));
            NDArray output = trainer.evaluate(new NDList(input)).singletonOrThrow();
            return toDoubleArray(output.toFloatArray());
        }
    }

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
            double[] gradient = toDoubleArray(input.getGradient().toFloatArray());
            collector.zeroGradients();
            return gradient;
        }
    }

    private void trainBatch() {
        ensureOpen();
        float[] inputs = Arrays.copyOf(inputBatch, batchSize * inputDim);
        float[] outputs = Arrays.copyOf(outputBatch, batchSize * outputDim);

        try (NDManager batchManager = manager.newSubManager()) {
            NDArray input = batchManager.create(inputs, new Shape(batchSize, inputDim));
            NDArray output = batchManager.create(outputs, new Shape(batchSize, outputDim));
            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(input)
                            .optLabels(output)
                            .setSampling(batchSize, false)
                            .build();
            try (Batch batch = trainer.iterateDataset(dataset).iterator().next()) {
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
            }
            batchSize = 0;
        } catch (IOException | TranslateException e) {
            throw new IllegalStateException("failed to train MLP batch", e);
        }
    }

    private static SequentialBlock createBlock(int outputDim) {
        return new SequentialBlock()
                .add(Linear.builder().setUnits(256).build())
                .add(Activation.sigmoidBlock())
                .add(Linear.builder().setUnits(128).build())
                .add(Activation.sigmoidBlock())
                .add(Linear.builder().setUnits(128).build())
                .add(Activation.sigmoidBlock())
                .add(Linear.builder().setUnits(outputDim).build());
    }

    private static DefaultTrainingConfig setupTrainingConfig() {
        Adam optimizer =
                Adam.builder()
                        .optLearningRateTracker(Tracker.fixed(LEARNING_RATE))
                        .build();
        return new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(optimizer)
                .addTrainingListeners(TrainingListener.Defaults.logging());
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
