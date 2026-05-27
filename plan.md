Goal: Fast inference for fixed-effects panel quantile regression (FEQR) with common shocks and tens of millions of observations
- So basically combine [Chiang, Galvao, and Wei (2026)](https://arxiv.org/abs/2602.19201) and [Lee, Liao, Seo, and Shin (2023)](https://arxiv.org/abs/2209.14502). 

References I've used but haven't been able to explicitly cite yet include [Koenker and Bassett (1978)](https://doi.org/10.2307/1913643) and [Polyak and Juditsky (1992)](https://doi.org/10.1137/0330046).




# Key objects

## FEQR objective $Q_{NT}(\alpha,\beta)$ and time-block decomposition $q_t(\alpha,\beta)$

[Chiang, Galvao, and Wei (2026)](https://arxiv.org/abs/2602.19201) ([HTML version](https://arxiv.org/html/2602.19201v2)) define, in their equation (4), the "standard Koenker-type FEQR estimator ([Koenker, 2004](https://doi.org/10.1016/j.jmva.2004.05.006)) without regularisation" like so:
$$
(\boldsymbol{\hat{\alpha}}(\tau),\hat\beta(\tau))
=
\arg\min_{\{\alpha_i\}_{i=1}^N,\beta}
\frac{1}{NT}
\sum_{i=1}^N\sum_{t=1}^T
\rho_\tau\!\left(Y_{it}-\alpha_i-X_{it}'\beta\right),
$$ {#eq-CGW-4}
"where $\rho_\tau(u)=u\{\tau-\mathbf 1(u<0)\}$ is the check loss function" (p. 8) for a fixed quantile index $\tau\in(0,1)$.^[Note how this is the same FEQR estimator given in equation (2), p. 3 of [Galvao and Wang (2015)](https://doi.org/10.1016/j.jmva.2014.08.007), but with $n$ used for the cross-sectional dimension instead of $N$.]^[Also note how the broader longitudinal quantile regression framework comes from [Koenker (2004)](https://doi.org/10.1016/j.jmva.2004.05.006), whereas its large-panel asymptotic theory was developed by [Kato, Galvao, and Montes-Rojas (2012)](https://doi.org/10.1016/j.jeconom.2012.02.007).]

Given panel observations $\{(Y_{it},X_{it}):i=1,\ldots,N,\ t=1,\ldots,T\}$, individual effects $\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_N)'$, and common slope $\beta\in\mathbb R^p$, define the FEQR objective:
$$
Q_{NT}(\alpha,\beta)
:=
\frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^T
\rho_\tau\!\left(Y_{it}-\alpha_i-X_{it}'\beta\right).
$$ {#eq-QNT}

Alternatively, we can first define the per-time-block objective:
$$
q_t(\alpha,\beta)
:=
\frac{1}{N}\sum_{i=1}^N
\rho_\tau\!\left(Y_{it}-\alpha_i-X_{it}'\beta\right).
$$ {#eq-qt}
Grouping terms with a shared $t$ thus lets us rewrite @eq-QNT as the equivalent:
$$
Q_{NT}(\alpha,\beta)=\frac{1}{T}\sum_{t=1}^T q_t(\alpha,\beta).
$$ {#eq-time-block-decomposition}

The above formulation of $q_t$, as an algebraic time-block decomposition of the FEQR objective, follows from p. 3 of [Lee, Liao, Seo, and Shin (2023)](https://arxiv.org/abs/2209.14502). They first define their cross-sectional population objective:
$$
Q(\beta):=E[q(\beta,Y_i)],
$$
where the individual loss function $q(\cdot)$ is defined as the check function
$$
q(\beta,Y_i):=(y_i-x_i'\beta)\{\tau-\mathbf{1}(y_i-x_i'\beta\le0)\}.
$$
Their sample estimator is thus:
$$
\hat\beta_n
:=
\arg\min_{\beta\in\mathbb R^d}
\frac{1}{n}\sum_{i=1}^n q(\beta,Y_i).
$$ {#eq-LLSS-2}

My proposed $q_t(\alpha,\beta)$ is, effectively, the panel analog of Lee et al.’s single-observation loss contribution $q(\beta,Y_i)$, except that one time block contains the whole cross section $i=1,\ldots,N$. My $q_t(\cdot)$ also contains an additional $1/N$ normalization. This will eventually make it so that $Q_{NT}=T^{-1}\sum_t q_t$ exactly matches Chiang et al.'s normalization $(NT)^{-1}\sum_{i,t}$, while also ensuring that the $\beta$-score per time block is naturally order one in the limit under common shocks.

**Qualifier:** For the full $(\boldsymbol\alpha,\beta)$, one has to remember that the $\alpha_i$-coordinate of $q_t$ affects only unit $i$; the $\alpha$-subgradient and the $\beta$-subgradient live on different effective scales. We'll likely thus forego a direct $(N+p)$-dimensional stochastic approximation theorem and instead use either a profiled slope recursion or an asymptotic equivalence argument for now.



Further reasons for defining the time-block decomposition $q_t(\cdot)$ can be found in [Boyd, Mutapcic, and Duchi (2018)](https://web.stanford.edu/class/ee364b/lectures/stoch_subgrad_notes.pdf), who on p. 13 discuss stochastic subgradient methods for large datasets and use an objective of the form
$$
f(x)=\frac{1}{m}\sum_{i=1}^m F(x;(a_i,b_i)).
$$
They then describe the stochastic subgradients obtained by subsampling terms. Instead of the subsampled term being the individual observation $(i,t)$, however, we choose the time block $F_t(\alpha, \beta)=q_t(\alpha,\beta)$ instead. This is because of the DGP assumptions we'll inherit from Chiang et al.: observations with the same $t$ will share $B_t$ under common shocks, meaning the cross-section should be kept together when approximating independent sampling.