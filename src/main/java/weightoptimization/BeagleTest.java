package weightoptimization;

import beagle.Beagle;
import beagle.BeagleFactory;

public class BeagleTest {
    Beagle beagle;

    static void main() {
        new BeagleTest();
    }

    public BeagleTest() {
        int tipCount = 4;
        int nodeCount = 7;
        int internalNodeCount = 3;

        int compactPartialsCount = tipCount;
        int stateCount = 4;
        int patternCount = 2;

        int eigenCount = 1;
        int categoryCount = 1;

        BufferIndexHelper partialBufferHelper = new BufferIndexHelper(nodeCount, tipCount);
        BufferIndexHelper eigenBufferHelper = new BufferIndexHelper(eigenCount, 0);
        BufferIndexHelper matrixBufferHelper = new BufferIndexHelper(nodeCount, 0);
        BufferIndexHelper scaleBufferHelper = new BufferIndexHelper(internalNodeCount, 0);

        int[] resourceList = null;
        int preferenceFlags = 2048;
        int requirementFlags = 0;

        this.beagle = BeagleFactory.loadBeagleInstance(
                tipCount,
                partialBufferHelper.getBufferCount(),
                compactPartialsCount,
                stateCount,
                patternCount,
                eigenBufferHelper.getBufferCount(),
                matrixBufferHelper.getBufferCount(),
                categoryCount,
                scaleBufferHelper.getBufferCount(),
                resourceList,
                preferenceFlags,
                requirementFlags
        );

        // set states


    }

    public class BufferIndexHelper {
        /**
         * @param maxIndexValue the number of possible input values for the index
         * @param minIndexValue the minimum index value to have the mirrored buffers
         */
        BufferIndexHelper(int maxIndexValue, int minIndexValue) {
            this.maxIndexValue = maxIndexValue;
            this.minIndexValue = minIndexValue;

            offsetCount = maxIndexValue - minIndexValue;
            indexOffsets = new int[offsetCount];
            storedIndexOffsets = new int[offsetCount];
        }

        public int getBufferCount() {
            return 2 * offsetCount + minIndexValue;
        }

        void flipOffset(int i) {
            if (i >= minIndexValue) {
                indexOffsets[i - minIndexValue] = offsetCount - indexOffsets[i - minIndexValue];
            } // else do nothing
        }

        public int getOffsetIndex(int i) {
            if (i < minIndexValue) {
                return i;
            }
            return indexOffsets[i - minIndexValue] + i;
        }

        void getIndices(int[] outIndices) {
            for (int i = 0; i < maxIndexValue; i++) {
                outIndices[i] = getOffsetIndex(i);
            }
        }

        void storeState() {
            System.arraycopy(indexOffsets, 0, storedIndexOffsets, 0, indexOffsets.length);

        }

        void restoreState() {
            int[] tmp = storedIndexOffsets;
            storedIndexOffsets = indexOffsets;
            indexOffsets = tmp;
        }

        private final int maxIndexValue;
        private final int minIndexValue;
        private final int offsetCount;

        private int[] indexOffsets;
        private int[] storedIndexOffsets;

    }
}
