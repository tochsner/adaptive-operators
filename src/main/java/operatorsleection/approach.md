### Optimization through operator selection (09.09.2026)

We introduce a simple strategy to improve operator selection.

Given ordered operators with integer weights $w_k$ and cumulative weights $c_k$, we enlarge the state space $X$ as $\tilde{X} = X \times \{1 \ldots  C = \sum_k{w_k} \}$. In state $(\cdot, c)$, we only use the operator $k'$ with $c_{k'-1} < c \leq c_{k'}$. The main idea is to design the kernel to stay in state $k$ for a while before switching. This might improve mixing when there is non-ridge dependencies between parameter blocks, as it effectively samples the current parameter block conditioned on the rest.

One step works as follows:

1. We are currently in state $(x, k)$.
2. We advance $k' = (k \mod C$) + 1.
3. Propose $\tilde{x} \sim op_k(x)$.
4. Set $x' = \tilde{x}$ on acceptance, $x'=x$ otherwise.
5. Set the new state as $(x', k')$.

Given numerical weights, we can turn them into integer weights by rounding them to some level of precision. We can also set a maximum dwell time and repeat the pruned cycle.