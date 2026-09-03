### A new augmented tree space (31.08.2026)

We augment the Felsenstein tree distribution to improve mixing.

We have $n$ taxa with sampling times $\rho(i)$ for taxa $i$.

*Augmented State:*

The state of the tree with $n$ taxa is extended to a $n \times n$ symmetric non-negative distance matrix $D$. The tree can be recovered by running average-linkage clustering on the distance matrix.

We will now propose a distribution on $D$ which is a proper auxiliary distribution for $T$:

$$
p(D, \theta) = \pi(T(D), \theta) p(D | T).
$$
If we marginalize out all possible distance matrices leading to a tree $T$, we will recover $\pi(T, \theta)$.

*Average-linkage Clustering:*

Given the distance matrix, we perform average-linkage clustering (UPGMA).

We define the cluster distance of two clusters $A$ and $B$ as

$$
d(A, B) = \frac{1}{|A||B|}\sum_{a \in A, b \in B}{d(a, b)}.
$$
We start with the clusters $P(0) = \{ \{1\}, \ldots \{n\} \}$ and then iteratively merge clusters. $P(r)$ denotes the clusters after $r$ merges.

In the $r$-th merge step, we obtain a new cluster
$$
X(r) = A(r) \cup B(r)
$$
such that $P(r) = P(r-1) \cup \{X(r)\} \setminus \{A(r), B(r)\}$. $A(r)$ and $B(r)$ are defined as the two closest clusters:
$$
A(r), B(r) = \arg\min_{X \neq Y\in P(r-1)}{d(X, Y)}.
$$

We denote the merge distance with
$$
d(r) =d(A(r), B(r)).
$$

*Dealing with heterochronous sampling:*

UPGMA assumes homochronous sampling, it only produces ultrametric trees. However, we can treat the distances in $D$ as extended distances to taxa mapped to the present and then prune the extant branches after tree reconstruction. Some $D$ will have a average-linkage tree with negative branch lengths. We simply reject these trees and set their density to $0$.

*Slacks:*

For every cluster pair in $P(r-1)$, we define its *slack* as
$$
s_{r}(A, B) = d(A, B) - d(r).
$$

This can be interpreted as how far away that cluster pair is from being merged in that round. By definition, $s_r(A, B) \geq 0$ with $s_r(A(r), B(r)) = 0$.

In round $r$, for clusters $V \in P(r)$ and $X(r) = A(r) \cup B(r)$, we have
$$
\begin{aligned}
d(V, X) &= \frac{|A|d(V, A) + |B|d(V, B)}{|X|} \\
&= \frac{|A|(s_{r}(V, A) + d(r)) + |B|(s_{r}(V, B) + d(r))}{|X|} \\
&= \frac{|A|s_{r}(V, A) + |B|s_{r}(V, B)}{|X|} + d(r)
\end{aligned}
$$
and thus
$$
\begin{aligned}
|A|s_{r}(V, A) + |B|s_{r}(V, B) &= |X| (d(V, X) - d(r)) \\
&= |X| (d(V, X) - d(r) - d(r+1) + d(r+1)) \\
&= |X| (s_{r+1}(V, X) + d(r+1) - d(r)) \\
&= |X| (s_{r+1}(V, X) + \Delta_r)
\end{aligned}
$$
with $\Delta_r = d(r+1) - d(r)$.

For two clusters $U, V \in P(r)$ that are not affected by the merging, we have
$$
s_r(U, V) = d(U, V) - d(r) = s_{r+1}(U, V) + \Delta_r.
$$

*Generative process:*

Given a tree topology $T$ and the intervals $\Delta$, we can sample the slacks and a distance matrix $D$ backwards from $r = n - 2$ to $r=1$.

Concretely, for every pair $U, V \in P(r-1)$, we do one of the following:

If $U$ and $V$ get merged, we set $s_r(U, V) = 0$.

If neither $U$ nor $V$ gets merged, we set $s_r(U, V) = s_{r+1}(U, V) + \Delta_r$.

If $U$ but not $V$ gets merged (with $U \cup B = X$), the following equation holds:
$$
|U|s_{r}(V, U) + |B|s_{r}(V, B) = |X| (s_{r+1}(V, X) + \Delta_r).
$$
We can sample
$$
p_r(U, V) \sim Beta(\cdot,\cdot)
$$
and set
$$
s_r(U, V) = \frac{p_r(U, V) |X| (s_{r+1}(V, X) + \Delta_r)}{|U|},
$$
and
$$
s_r(B, V) = \frac{(1 - p_r(U, V)) |X| (s_{r+1}(V, X) + \Delta_r)}{|B|}.
$$
This process will generate valid $s$ that are nonnegative. Applying average-linkage on the corresponding distance matrix will recover the tree again.

*Beta parameters:*

How should we parameterized the fractions?

We have
$$
|X| (s_{r+1}(V, X) + \Delta_r) = |A|s_{r}(V, A) + |B|s_{r}(V, B)
$$
$p_r(V, A)$ determines how close $V$ is to $A$ compared to $B$. We can look at a cheap pairwise-distance score induced by the expected substitutions $d_{SM}$ and the substitution rate $r_{SM}$:
$$
S_r(V, A) = \frac{1}{|V||A|}\sum_{v\in V, a \in A}{d_{SM}(v, a) / r_{SM}}.
$$

For example, we can set

$$
\begin{aligned}
p_r(V, A) \sim Beta(c\frac{|A|S_r(V, A)}{|A|S_r(V, A) + |B|S_r(V, B)}, c\frac{|B|S_r(V, B)}{|A|S_r(V, A) + |B|S_r(V, B)}).
\end{aligned}
$$

$c > 0$ is a concentration hyperparameter.

*Density:*

The augmented state of the MCMC is the distance matrix $D$. The tree $T$ (including branch lengths) and the $\frac{(n-1)(n-2)}{2}$ random variables for each fraction $p_r(V, U)$ are deterministic functions of $D$. The augmented probability density is thus

$$
p(D, \theta) = \pi(T(D), \theta) \cdot \mathbf{1}_{\mathcal{T}}(T(D)) \cdot \left( \prod_p{Beta(p; \cdot, \cdot) } \right) \cdot J(D)^{-1}
$$
where $\mathcal{T}$ is the set of valid trees with positive branch lengths and the Jacobian is defined as
$$
\begin{aligned}
J(D) = \prod_{r=1}^{n-2} \prod_{V \in P(r) \setminus \{ X(r) \}} \frac{|X(r)|^2\left(s_{r+1}(V, X(r))+\Delta_r\right)}{|A(r)||B(r)|}.
\end{aligned}
$$

*Possible algorithms:*

We can use this conditional density in a few different ways:

1. Gibbs refresh sampler: We can treat $T$ as the state and introduce a new proposal. The proposal samples $D$, perturbs it to get $D'$, and runs UPGMA on $D'$ to get a proposal $T'$.
2. Data augmentation: We augment the state to $D$ and use the augmented distribution to perform MCMC in this augmented state.
3. Local augmentation: only perturb the distances for the distances of a randomly selected clade of size $k$. This counters the curse of high dimensionality.
4. Apply HMC on the auxiliary tree distribution / use a slice sampler on $D$.

*Note (03.09.2026): centering the Beta weights on the merge distance*

The parameterization above uses the raw scores $S_r(V, A)$ and $S_r(V, B)$. Both contain the shared component $d(r)$, since every path from $V$ to $A$ or $B$ passes through the merge of $A$ and $B$, so for deep nodes the ratio collapses towards $|A| / |X|$ and carries almost no information from the data.

The quantity $p_r(V, A)$ actually models is the fraction of the excess over the merge distance. If $D$ were the data-implied distance matrix $D^*$, we would have
$$
p_r^*(V, A) = \frac{|A| (d^*(V, A) - d(r))}{|A| (d^*(V, A) - d(r)) + |B| (d^*(V, B) - d(r))}.
$$
We therefore now use the centered weights
$$
p_r(V, A) \sim Beta(c\, m, c\, (1 - m)), \quad m = \frac{|A| (S_r(V, A) - d(r))_+}{|A| (S_r(V, A) - d(r))_+ + |B| (S_r(V, B) - d(r))_+},
$$
with $(x)_+ = \max(x, 0)$. For the units to match, $S_r$ is measured in the units of $D$: each pairwise substitution distance is divided by the substitution rate and extended to the present by adding the two sampling times, $d^*(i, j) = d_{SM}(i, j) / r_{SM} + \rho(i) + \rho(j)$. When both excesses are clipped to zero (the tree is deeper than the data suggests), $m$ falls back to the no-information value $|A| / |X|$. $m$ is clamped to $[0.1, 0.9]$ so that neither Beta shape can collapse to zero, which would put the slack exactly at zero and create ties in the clustering.

The sampler and the conditional density must use the same $m$; both call the same helper.
