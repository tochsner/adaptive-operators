package schedule;

import beast.base.inference.Operator;
import weightoptimization.ESSWeightAdaptationOperator;

import java.io.PrintStream;

/**
 * Dual averaging schedule that, on top of the usual operator statistics, reports the weights chosen
 * by every {@link ESSWeightAdaptationOperator} it contains.
 *
 * <p>{@code OperatorSchedule.prettyPrintOperator} is static and so cannot be specialised; the
 * additional tables are printed after the standard one instead.
 */
public class ESSWeightOperatorSchedule extends DualAveragingOperatorSchedule {

    @Override
    public void showOperatorRates(PrintStream out) {
        super.showOperatorRates(out);

        for (Operator operator : this.operators) {
            if (operator instanceof ESSWeightAdaptationOperator weightOperator) {
                this.printWeights(weightOperator, out);
            }
        }
    }

    /**
     * Recomputes the selection probabilities from the current operator weights.
     */
    public void reweight() {
        this.reweightOperators();
    }

    private void printWeights(ESSWeightAdaptationOperator weightOperator, PrintStream out) {
        String status = weightOperator.isAdapted() ? "" : " (never adapted, initial weights in use)";

        out.println("Weights of " + weightOperator.getName() + status);
        out.println(weightOperator.formatWeightTable());
    }
}
