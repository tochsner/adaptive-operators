### Exact local transport map (22.06.2026)

*Issue:* The cube parameterization for trees is continuous, but can be highly non-monotonous. This renders local kernels less effective at exploring the tree space.

*Idea:* Use a transport map informed by sequences to operate on a nicer space.

*Simplification and localization:*

We start with a tree $T$. We condition on a cube compatible with $T$ and and look at $k=4$ neighboring taxa. Hopefully, this leads to four taxa which are locally related, with the majority of the uncertainty regarding their local arrangement.

We denote the distances of these four taxa with $d_{1:3}$.

We want to sample from the potentially multimodal posterior $p(d)$. We simplify even further by approximating the posterior using the likelihood:

$$
p(d) \approx \frac{l(d)}{C}
$$

with some normalizing constant $C$.

We approximate $l(d)$ with the Felsenstein likelihood of the subtree consisting of the four taxa and a JC69 substitution model with the current clock rate. We call this approximate (normalized) likelihood $l_F(d)$.

*Transport maps:*

We want to learn a transport map $T(d): d \mapsto z$ such that $z$ follows a nice unimodal distribution if $d$ follows $l_F(d)$. Using a triangular transport map with the Knothe–Rosenblatt rearrangement, we obtain:

$$
\begin{align}
T_1(d) &= G^{-1}\left(F\left(d_1\right)\right)  \\
T_2(d) &= G^{-1}\left(F\left(d_2 | d_1 \right) |z_1 \right) \\
T_3(d) &= G^{-1}\left(F\left(d_3 | d_1, d_2 \right) |z_1, z_2 \right).
\end{align}
$$
where $G$ is the conditional CDF of the standard multivariate Gaussian and $F$ is the conditional CDF of $l_F(d)$. We can compute $F$ using numerical integration, which at worst is a two-dimensional integral.

*Operator:*

A move using the transport map works as follows:

1. We start with the old $d$ and compute $z=T(d)$ using numerical integration.
2. In $z$-space, we add Gaussian noise with a learned scale to obtain $z^\ast = z + \gamma$.
3. We use numerical integration and univariate root-finding to map $z^\ast$ back to $d$-space: $d^\ast = T^{-1}(z^\ast)$.
4. The log acceptance probability is as follows:

$$
\begin{align}
\log \mathrm{\alpha} &= \log \pi(d^\ast) - \log \pi(d) + \log |\det \nabla T(d)| - \log |\det \nabla T(d^*)| \\
&= \log{\pi(d^\ast)} - \log{\pi(d)} + (\log l_F(d) - \log \varphi(z)) - (\log l_F(d^\ast) - \log \varphi(z^\ast)) \\
&= \log{\pi(d^\ast)} - \log{\pi(d)} + \log l_F(d) - \log l_F(d^\ast) - \log \varphi(z) + \log \varphi(z^\ast).
\end{align}
$$
