package mcmc;

import beast.base.core.BEASTObject;
import beast.base.core.Log;
import beast.base.core.Loggable;
import beast.base.inference.Logger;
import beast.base.inference.Operator;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * MCMC driver that ends the chain as soon as every logged numeric quantity has an effective sample size (ESS) of at
 * least {@value #TARGET_ESS}, instead of waiting for the full chain length.
 *
 * <p>For every logger, a line of the log is built every {@code logEvery} samples by calling {@code log} on all of its
 * loggables, and its numeric items are fed to a {@link ThinnedEssTracker}, which discards the first 10% of the trace
 * as burn-in. Items that cannot be parsed as numbers, such as trees, are ignored, and so are items that have never
 * changed, as they have no meaningful ESS. The chain length acts as an upper bound.
 */
public class ESSStoppingMCMC extends JumpMCMC {

    private static final double TARGET_ESS = 200;

    private List<LoggerTracker> loggerTrackers;

    @Override
    protected Operator propagateState(final long sampleNr) {
        final Operator operator = super.propagateState(sampleNr);

        if (sampleNr >= 0 && this.recordLoggedValues(sampleNr) && this.hasConverged()) {
            Log.info.println("\nAll logged parameters reached an ESS of " + TARGET_ESS + " at sample " + sampleNr + ", stopping the chain.");

            // the main loop ends after this step, which still takes care of storing the final state
            this.chainLength = sampleNr;
        }

        return operator;
    }

    /** Feeds the current values to the trackers of all loggers that are due, and returns whether any were. */
    private boolean recordLoggedValues(final long sampleNr) {
        if (this.loggerTrackers == null) {
            this.loggerTrackers = this.createLoggerTrackers();
        }

        boolean recorded = false;
        for (final LoggerTracker loggerTracker : this.loggerTrackers) {
            if (sampleNr % loggerTracker.logEvery == 0) {
                loggerTracker.record(sampleNr);
                recorded = true;
            }
        }
        return recorded;
    }

    private List<LoggerTracker> createLoggerTrackers() {
        final List<LoggerTracker> trackers = new ArrayList<>();
        for (final Logger logger : this.loggers) {
            trackers.add(new LoggerTracker(logger));
        }
        return trackers;
    }

    private boolean hasConverged() {
        boolean anyTracked = false;

        for (final LoggerTracker loggerTracker : this.loggerTrackers) {
            for (final ColumnTracker column : loggerTracker.columns) {
                if (!column.hasVaried) continue;

                if (!column.tracker.hasReached(TARGET_ESS)) return false;
                anyTracked = true;
            }
        }

        return anyTracked;
    }

    /** The trackers of all numeric items of a single logger. */
    private static class LoggerTracker {

        private final int logEvery;
        private final List<Loggable> loggables = new ArrayList<>();
        private final List<ColumnTracker> columns = new ArrayList<>();

        LoggerTracker(final Logger logger) {
            this.logEvery = logger.everyInput.get();

            for (final BEASTObject loggable : logger.loggersInput.get()) {
                this.loggables.add((Loggable) loggable);
            }
        }

        void record(final long sampleNr) {
            final String[] items = this.buildLogLine(sampleNr).split("\\s+");

            // the items of a line can change in number if non-numeric items such as trees are mixed in,
            // so they are tracked by position and only numeric ones are used
            for (int i = 0; i < items.length; i++) {
                final double value = parseNumber(items[i]);
                if (Double.isNaN(value)) continue;

                while (this.columns.size() <= i) {
                    this.columns.add(new ColumnTracker());
                }
                this.columns.get(i).record(value);
            }
        }

        private String buildLogLine(final long sampleNr) {
            final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            final PrintStream out = new PrintStream(bytes);

            for (final Loggable loggable : this.loggables) {
                loggable.log(sampleNr, out);
            }
            out.flush();

            return bytes.toString().trim();
        }

        private static double parseNumber(final String item) {
            try {
                return Double.parseDouble(item);
            } catch (NumberFormatException e) {
                return Double.NaN;
            }
        }
    }

    /** The ESS tracker of a single logged item, which also remembers whether the item has ever changed. */
    private static class ColumnTracker {

        private final ThinnedEssTracker tracker = new ThinnedEssTracker(1);

        private double firstValue = Double.NaN;
        private boolean hasVaried = false;

        void record(final double value) {
            if (Double.isNaN(this.firstValue)) {
                this.firstValue = value;
            } else if (value != this.firstValue) {
                this.hasVaried = true;
            }

            this.tracker.record(value);
        }
    }
}
