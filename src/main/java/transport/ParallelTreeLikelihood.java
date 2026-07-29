package transport;

import beast.base.evolution.tree.Tree;
import beast.base.spec.evolution.likelihood.ThreadedTreeLikelihood;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiFunction;

public class ParallelTreeLikelihood {

    int numInstances;
    ThreadedTreeLikelihood[] treeLikelihoods;
    Tree[] trees;
    boolean isSetup = false;

    public ParallelTreeLikelihood(int numInstances) {
        if (numInstances < 1) {
            throw new IllegalArgumentException("numInstances must be at least 1.");
        }

        this.numInstances = numInstances;
        this.treeLikelihoods = new ThreadedTreeLikelihood[numInstances];
        this.trees = new Tree[numInstances];
    }

    public void setup(ThreadedTreeLikelihood treeLikelihood, Tree tree) {
        Objects.requireNonNull(treeLikelihood, "treeLikelihood must not be null.");
        Objects.requireNonNull(tree, "tree must not be null.");

        for (int i = 0; i < this.numInstances; i++) {
            this.trees[i] = tree.copy();

            this.treeLikelihoods[i] = new ThreadedTreeLikelihood();
            this.treeLikelihoods[i].setID(treeLikelihood.getID() + ".parallel." + i);
            this.treeLikelihoods[i].dataInput.setTypedValue(treeLikelihood.dataInput.get(), this.treeLikelihoods[i]);
            this.treeLikelihoods[i].treeInput.setTypedValue(this.trees[i], this.treeLikelihoods[i]);
            this.treeLikelihoods[i].siteModelInput.setTypedValue(treeLikelihood.siteModelInput.get(), this.treeLikelihoods[i]);
            this.treeLikelihoods[i].branchRateModelInput.setTypedValue(treeLikelihood.branchRateModelInput.get(), this.treeLikelihoods[i]);
            this.treeLikelihoods[i].useAmbiguitiesInput.setTypedValue(treeLikelihood.useAmbiguitiesInput.get(), this.treeLikelihoods[i]);
            this.treeLikelihoods[i].rootFrequenciesInput.setTypedValue(treeLikelihood.rootFrequenciesInput.get(), this.treeLikelihoods[i]);
            this.treeLikelihoods[i].setInputValue("scaling", treeLikelihood.getInputValue("scaling"));
            this.treeLikelihoods[i].maxNrOfThreadsInput.setTypedValue(1, this.treeLikelihoods[i]);
            this.treeLikelihoods[i].initAndValidate();
        }

        this.isSetup = true;
    }

    public double[] evaluate(List<? extends BiFunction<ThreadedTreeLikelihood, Tree, Double>> evaluators) {
        if (!this.isSetup) {
            throw new IllegalStateException("setup must be called before evaluate.");
        }
        Objects.requireNonNull(evaluators, "evaluators must not be null.");

        double[] results = new double[evaluators.size()];
        if (evaluators.isEmpty()) {
            return results;
        }

        int workerCount = Math.min(this.numInstances, evaluators.size());
        ExecutorService executor = Executors.newFixedThreadPool(workerCount);
        try {
            List<Callable<Void>> workers = new ArrayList<>(workerCount);
            for (int worker = 0; worker < workerCount; worker++) {
                final int workerIndex = worker;
                workers.add(() -> {
                    ThreadedTreeLikelihood treeLikelihood = this.treeLikelihoods[workerIndex];
                    Tree tree = this.trees[workerIndex];

                    for (int evaluatorIndex = workerIndex; evaluatorIndex < evaluators.size(); evaluatorIndex += workerCount) {
                        BiFunction<ThreadedTreeLikelihood, Tree, Double> evaluator = evaluators.get(evaluatorIndex);
                        results[evaluatorIndex] = Objects.requireNonNull(evaluator, "evaluator must not be null.")
                                .apply(treeLikelihood, tree);
                    }

                    return null;
                });
            }

            List<Future<Void>> futures = executor.invokeAll(workers);
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel tree likelihood evaluation was interrupted.", e);
        } catch (Exception e) {
            throw new RuntimeException("Parallel tree likelihood evaluation failed.", e);
        } finally {
            executor.shutdown();
        }

        return results;
    }

}
