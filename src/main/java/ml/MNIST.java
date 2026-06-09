package ml;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;


public class MNIST {
    static void main() throws IOException, TranslateException {
        Engine.debugEnvironment();

        // Construct neural network
        Block block =
                new Mlp(
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        Mnist.NUM_CLASSES,
                        new int[] {128, 64});

        try (Model model = Model.newInstance("mlp", "PyTorch")) {
            model.setBlock(block);

            // get training and validation dataset
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN);
            RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST);

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig();

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                /*
                 * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                EasyTrain.trainBatch(trainer, trainer.iterateDataset(trainingSet).iterator().next());
                EasyTrain.fit(trainer, 1_0, trainingSet, validateSet);

                trainer.getTrainingResult();
            }
        }
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage)
            throws IOException {
        Mnist mnist =
                Mnist.builder()
                        .optUsage(usage)
                        .optManager(NDManager.newBaseManager("PyTorch"))
                        .setSampling(32, true)
                        .build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }

    private static DefaultTrainingConfig setupTrainingConfig() {
        String outputDir = "/";
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }
}
