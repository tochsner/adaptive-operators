package upgma;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

import java.util.Arrays;

/**
 * Chooses which taxa's pairwise distances get perturbed in a {@link UpgmaRefreshOperator} proposal,
 * and in clade mode reconciles the reclustered tree with that choice.
 *
 * <p>{@link #select()} redraws the affected set at the start of every proposal. In clade mode the
 * choice depends on the current tree, so {@link #reconcile} has to check that the reverse move
 * could make the same choice.
 */
public final class UpgmaTaxonSelection {

    /** How the set of perturbed distances is chosen in every proposal. */
    public enum Mode {
        /** Noise on every distance among a uniformly random subset of {@code maxSize} taxa. */
        UNIFORM,
        /** Noise on every distance. */
        ALL,
        /** Noise on every distance inside a random non-root clade of at most {@code maxSize} taxa. */
        CLADE;

        /** Parses a case-insensitive mode name. */
        public static Mode parse(String name) {
            try {
                return Mode.valueOf(name.trim().toUpperCase());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("selectionMode must be one of uniform, all or clade, got: " + name);
            }
        }
    }

    private final Mode mode;
    private final Tree tree;
    private final int n;

    private final int maxSize;

    private final boolean[] affected;
    private int affectedCount;

    // clade mode: the selected clade's node number in the current tree, and how many non-root clades
    // were eligible for selection, which the reverse move has to match in the Hastings ratio
    private int selectedClade;
    private int eligibleCladeCount;

    // workspaces indexed by tree node number
    private final int[] leafCounts;

    // workspaces over cluster ids 0 .. 2n-2 of a clustering result
    private final int[] clusterMembers;
    private final int[] clusterSize;
    private final int[] oldNodeOfCluster;

    public UpgmaTaxonSelection(Mode mode, Tree tree, int maxSize) {
        this.mode = mode;
        this.tree = tree;
        this.n = tree.getLeafNodeCount();
        this.maxSize = maxSize > 0 ? Math.min(maxSize, this.n) : this.n;

        this.affected = new boolean[this.n];
        this.leafCounts = new int[2 * this.n - 1];
        this.clusterMembers = new int[2 * this.n - 1];
        this.clusterSize = new int[2 * this.n - 1];
        this.oldNodeOfCluster = new int[2 * this.n - 1];
    }

    public boolean isAffected(int i) {
        return this.affected[i];
    }

    /** Draws a new affected set from the current tree, according to the mode. */
    public void select() {
        Arrays.fill(this.affected, false);
        this.affectedCount = 0;

        switch (this.mode) {
            case ALL:
                this.selectAll();
                break;

            case UNIFORM:
                int[] permutation = Randomizer.shuffled(this.n);
                for (int i = 0; i < this.maxSize; i++) {
                    this.affected[permutation[i]] = true;
                }
                this.affectedCount = this.maxSize;
                break;

            case CLADE:
                this.selectClade();
                break;
        }
    }

    /**
     * Marks the leaves of a uniformly random non-root clade of at most {@link #maxSize} taxa.
     * Falls back to all taxa if there is no such clade (n < 3).
     */
    private void selectClade() {
        countLeaves(this.tree.getRoot(), this.leafCounts);

        int eligible = 0;
        for (int nr = this.n; nr < this.tree.getNodeCount(); nr++) {
            if (this.isEligible(nr)) {
                eligible++;
            }
        }
        this.eligibleCladeCount = Math.max(eligible, 1);

        if (eligible == 0) {
            this.selectAll();
            this.selectedClade = this.tree.getRoot().getNr();
            return;
        }

        int pick = Randomizer.nextInt(eligible);
        for (int nr = this.n; nr < this.tree.getNodeCount(); nr++) {
            if (this.isEligible(nr) && pick-- == 0) {
                this.selectedClade = nr;
                for (Node leaf : this.tree.getNode(nr).getAllLeafNodes()) {
                    this.affected[leaf.getNr()] = true;
                }
                this.affectedCount = this.leafCounts[nr];
                return;
            }
        }
    }

    private void selectAll() {
        Arrays.fill(this.affected, true);
        this.affectedCount = this.n;
    }

    private boolean isEligible(int nr) {
        return !this.tree.getNode(nr).isRoot() && this.leafCounts[nr] <= this.maxSize;
    }

    private static int countLeaves(Node node, int[] leafCounts) {
        int count = node.isLeaf() ? 1 : 0;
        for (Node child : node.getChildren()) {
            count += countLeaves(child, leafCounts);
        }
        leafCounts[node.getNr()] = count;
        return count;
    }

    /**
     * Reconciles a clustering of the perturbed distances with the current selection, before the tree
     * is modified. Returns the log ratio of the reverse to the forward selection probability, or
     * {@code 0} outside clade mode.
     *
     * <p>In clade mode a cluster that mixes members and non-members of the selected clade C without
     * containing all of C means C dissolved, so the reverse move could not select it and the
     * proposal is rejected with {@link Double#NEGATIVE_INFINITY}. Every other cluster outside C is
     * an existing clade of the current tree, and its merge height is snapped back to that node's
     * exact height so {@code TreeUtils.applyClustering} leaves the node untouched.
     */
    public double reconcile(UpgmaClustering.Result clustering) {
        if (this.mode != Mode.CLADE) {
            return 0.0;
        }
        if (!this.snapUnaffectedHeights(clustering)) {
            return Double.NEGATIVE_INFINITY;
        }
        return Math.log(this.eligibleCladeCount) - Math.log(this.countEligibleClusters());
    }

    private boolean snapUnaffectedHeights(UpgmaClustering.Result clustering) {
        for (int leaf = 0; leaf < this.n; leaf++) {
            this.clusterMembers[leaf] = this.affected[leaf] ? 1 : 0;
            this.clusterSize[leaf] = 1;
            this.oldNodeOfCluster[leaf] = leaf;
        }

        for (int merge = 0; merge < this.n - 1; merge++) {
            int cluster = this.n + merge;
            int left = clustering.mergeLeft()[merge];
            int right = clustering.mergeRight()[merge];

            int members = this.clusterMembers[left] + this.clusterMembers[right];
            int size = this.clusterSize[left] + this.clusterSize[right];
            this.clusterMembers[cluster] = members;
            this.clusterSize[cluster] = size;

            if (members == size) {
                // inside C (or C itself): its merge height legitimately changed, only track the node
                this.oldNodeOfCluster[cluster] = members == this.affectedCount ? this.selectedClade : -1;
                continue;
            }
            if (members > 0 && members < this.affectedCount) {
                return false;
            }

            int oldLeft = this.oldNodeOfCluster[left];
            int oldRight = this.oldNodeOfCluster[right];
            if (oldLeft < 0 || oldRight < 0) {
                throw new IllegalStateException("cluster outside the affected clade has a child inside it");
            }
            Node parent = this.tree.getNode(oldLeft).getParent();
            if (parent == null || parent != this.tree.getNode(oldRight).getParent()) {
                throw new IllegalStateException("cluster outside the affected clade is not a clade of the current tree");
            }

            this.oldNodeOfCluster[cluster] = parent.getNr();
            clustering.mergeHeight()[merge] = parent.getHeight();
        }
        return true;
    }

    /** Number of non-root clusters with at most {@link #maxSize} leaves; needs {@link #clusterSize} filled. */
    private int countEligibleClusters() {
        int eligible = 0;
        for (int merge = 0; merge < this.n - 2; merge++) {
            if (this.clusterSize[this.n + merge] <= this.maxSize) {
                eligible++;
            }
        }
        return Math.max(eligible, 1);
    }
}
