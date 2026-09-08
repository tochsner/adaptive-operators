"### Data-informed AR(1) perturbation (05.09.2026)

A drift-informed alternative to the symmetric Gaussian random walk on the selected distances. The
proposal is reversible with respect to a per-pair Gaussian surrogate of the pairwise likelihood, so
it drifts the distances towards the data while the Hastings correction stays closed form.

*Surrogate:*

For a pair $(i, j)$ with $L$ sites and $k$ mismatches, the JC69 pairwise log-likelihood in the
substitution distance $\delta$ is
$$
\ell(\delta) = k \log p(\delta) + (L - k) \log(1 - p(\delta)), \qquad p(\delta) = \tfrac{3}{4}\left(1 - e^{-4\delta/3}\right).
$$
Its maximizer is the JC69 distance $\hat\delta_{ij}$ and the observed Fisher information gives the
variance
$$
\hat\sigma^2_{ij} = \frac{p(1 - p)\, e^{8\hat\delta_{ij}/3}}{L}.
$$
$D$ lives in extended time units, so the surrogate for $d_{ij}$ is $N(\mu_{ij}, \sigma^2_{ij})$
with
$$
\mu_{ij} = \frac{\hat\delta_{ij}}{r} + t_i + t_j, \qquad \sigma^2_{ij} = \frac{\hat\sigma^2_{ij}}{r^2},
$$
where $t_i$ are the tip dates and $r$ is an adaptive rate. Every proposal observes the rate its
sampled matrix implies,
$$
\hat r(D) = \frac{\sum_{i < j} \hat\delta_{ij}}{\sum_{i < j} (d_{ij} - t_i - t_j)},
$$
and the operator's `optimize` folds $\log \hat r$ into a running mean with step $1 / k$. Within a
proposal $r$ is a constant. This sidesteps the substitution rate parameter, whose scale need not
match JC69 distances under other substitution models. $\hat\delta$ and $\hat\sigma^2$ are
computed once.

*Proposal:*

For every selected pair, with $\beta \in (0, 1]$ and $\epsilon \sim N(0, 1)$,
$$
d^\prime_{ij} = \mu_{ij} + \sqrt{1 - \beta^2}\,(d_{ij} - \mu_{ij}) + \beta\,\sigma_{ij}\,\epsilon.
$$
With $r$ fixed, this kernel leaves $N(\mu_{ij}, \sigma^2_{ij})$ invariant and is reversible with
respect to it, so
$$
\log\frac{q(D \mid D^\prime)}{q(D^\prime \mid D)} = \sum_{(i, j)\ \text{selected}} \left[ \log N(d_{ij}; \mu_{ij}, \sigma^2_{ij}) - \log N(d^\prime_{ij}; \mu_{ij}, \sigma^2_{ij}) \right].
$$
Unselected distances are copied through and contribute nothing.

*Adaptation:*

The rate moves only between proposals, under the same schedule as the scale factor. As with any
adaptive scheme, the adaptation has to diminish for the chain to stay ergodic. The $1 / k$ step
does, and BEAST's own tuning makes the same assumption.

*Why not discretized MALA:*

The Euler step $d^\prime = d + \alpha \nabla \log f(d) + \sqrt{2\alpha}\,\epsilon$ on a Gaussian is
an AR(1) with coefficient $1 - \alpha / \sigma^2$. It is unstable for $\alpha > \sigma^2$ and a
shared $\alpha$ across pairs with very different $\sigma^2$ makes some pairs oscillate while others
barely move. The exact update above is the same idea, stable for every $\beta$, preconditioned per
pair for free, and $\beta = 1$ is an independence proposal from the surrogate.

*Tuning:*

$\beta$ is the operator's scale factor, adapted on the logit scale and capped at $1$.

*Caveats:*

- Independent pairwise Gaussians are a composite likelihood, pairwise distances share branches.
  This costs acceptance rate, not correctness.
- JC69 distances can be biased under the actual substitution model, so $\mu$ can be off.
- Near-saturated pairs with $p \to 3/4$ get very large $\sigma^2$, which correctly weakens the
  drift for them.
"