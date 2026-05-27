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



## Score $\psi_{it}(\alpha,\beta)$ and subgradient objects $G_t^\beta(\alpha,\beta)$

### Recommended definitions

Define the residual

$$
r_{it}(\alpha,\beta)
:=Y_{it}-\alpha_i-X_{it}'\beta.
$$

Define the quantile score

$$
\psi_{it}(\alpha,\beta)
:=
\tau-
\mathbf 1\{Y_{it}\le \alpha_i+X_{it}'\beta\}
=
\tau-
\mathbf 1\{r_{it}(\alpha,\beta)\le0\}.
$$ {#eq-psi}

Define the per-time-block slope score

$$
G_t^\beta(\alpha,\beta)
:=
\frac{1}{N}\sum_{i=1}^N
\psi_{it}(\alpha,\beta)X_{it}.
$$ {#eq-Gt-beta}

Then the full-sample slope score is

$$
H_N^{(2)}(\alpha,\beta)
=
\frac{1}{T}\sum_{t=1}^T G_t^\beta(\alpha,\beta)
=
\frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^T
\psi_{it}(\alpha,\beta)X_{it}.
$$

### Important sign convention

The object $G_t^\beta$ is a **score**, not the literal gradient of $q_t$. Since

$$
q_t(\alpha,\beta)
=
\frac{1}{N}\sum_{i=1}^N \rho_\tau(Y_{it}-\alpha_i-X_{it}'\beta),
$$

a subgradient of $q_t$ with respect to $\beta$ is

$$
\partial_\beta q_t(\alpha,\beta)
\ni
-
\frac{1}{N}\sum_{i=1}^N
\psi_{it}(\alpha,\beta)X_{it}
=
-G_t^\beta(\alpha,\beta),
$$

up to the usual convention at zero residuals. Therefore a descent update can be written either as

$$
\beta_{s+1}
=
\beta_s-
\eta_s\{-G_{t_s}^\beta(\alpha_s,\beta_s)\}
=
\beta_s+
\eta_sG_{t_s}^\beta(\alpha_s,\beta_s),
$$

or, if one defines the literal subgradient

$$
\mathcal G_t^\beta(\alpha,\beta):=-G_t^\beta(\alpha,\beta),
$$

then

$$
\beta_{s+1}=\beta_s-\eta_s\mathcal G_{t_s}^\beta(\alpha_s,\beta_s).
$$

The previous informal response wrote updates in terms of subtracting $G_t^\beta$. That is sign-ambiguous unless $G_t^\beta$ is defined as the literal subgradient rather than the score. The cleaner convention for this paper is:

- $\psi_{it}=\tau-1\{r_{it}\le0\}$ is the **quantile score**.
- $G_t^\beta=N^{-1}\sum_i\psi_{it}X_{it}$ is the **slope score**.
- $-G_t^\beta$ is a **subgradient of the time-block loss** with respect to $\beta$.

This convention matches Chiang et al.’s notation for FEQR scores.

### Exact objects in the literature from which $\psi_{it}$ and $G_t^\beta$ follow

#### Chiang et al. (2026), Section 3.2 / page 13: $H_{Ni}^{(1)}$ and $H_N^{(2)}$

Chiang et al. define the subgradient/score objects on PDF page 13:

$$
H_{Ni}^{(1)}(\alpha_i,\beta)
=
\frac{1}{T}\sum_{t=1}^T
\left(\tau-
\mathbf 1\{Y_{it}\le \alpha_i+X_{it}'\beta\}\right),
$$

and

$$
H_N^{(2)}(\alpha,\beta)
=
\frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^T
\left(\tau-
\mathbf 1\{Y_{it}\le \alpha_i+X_{it}'\beta\}\right)X_{it}.
$$

Thus $\psi_{it}$ is simply the summand inside their $H_{Ni}^{(1)}$, and $G_t^\beta$ is the time-block summand whose average over $t$ equals their $H_N^{(2)}$:

$$
H_N^{(2)}(\alpha,\beta)=\frac{1}{T}\sum_{t=1}^T G_t^\beta(\alpha,\beta).
$$

**Status:** algebraic consequence of Chiang et al.’s score definitions.

#### Lee et al. (2023), Section 1.1 and Section 1.2

Lee et al. define the check loss

$$
q(\beta,Y_i)=(y_i-x_i'\beta)\{\tau-I(y_i-x_i'\beta\le0)\}
$$

and use a stochastic subgradient update on PDF page 5:

$$
\beta_i=\beta_{i-1}-\gamma_i\nabla q(\beta_{i-1},Y_i).
$$

For linear QR, a subgradient is

$$
\nabla q(\beta,Y_i)
= -x_i\{\tau-I(y_i-x_i'\beta\le0)\}
= x_i\{I(y_i-x_i'\beta\le0)-\tau\}.
$$

In the panel FEQR problem, replace one incoming cross-sectional observation $Y_i=(y_i,x_i)$ with one incoming time block

$$
\mathcal Y_t=\{(Y_{it},X_{it}):i=1,\ldots,N\}.
$$

Then Lee et al.’s stochastic subgradient architecture leads to a time-block subgradient of the form

$$
\nabla_\beta q_t(\alpha,\beta)=-G_t^\beta(\alpha,\beta).
$$

**Status:** algebraic consequence of Lee et al.’s subgradient update after changing the primitive stochastic unit from an observation to a time block.

### Is $G_t^\beta$ the most useful object?

For algorithm design, $G_t^\beta$ is useful but not yet sufficient.

For theorem work, a better slope score is likely the **orthogonalized Chiang score**

$$
M_t(\alpha,\beta,\gamma)
:=
\frac{1}{N}\sum_{i=1}^N
\psi_{it}(\alpha_i,
\beta)(X_{it}-\gamma_i),
$$ {#eq-orthogonal-time-block-score}

where Chiang et al. define

$$
\gamma_i=\frac{E[f_i(0\mid X_{i1})X_{i1}]}{f_i(0)}.
$$

This object is motivated by Chiang et al.’s asymptotic linear representation in Section A.2, equation (14), PDF page 27:

$$
\hat\beta-\beta_0
=
\Gamma_N^{-1}
\left(
\tilde H_N^{(2)}(\alpha_0,\beta_0)
-
\frac{1}{N}\sum_{i=1}^N
\gamma_i\tilde H_{Ni}^{(1)}(\alpha_{i0},\beta_0)
\right)
+o_p(T^{-1/2}).
$$ {#eq-CGW-14}

Heuristically, because

$$
\tilde H_N^{(2)}-\frac{1}{N}\sum_i\gamma_i\tilde H_{Ni}^{(1)}
\approx
\frac{1}{T}\sum_{t=1}^T
\frac{1}{N}\sum_{i=1}^N
E[\psi_{it}(\alpha_{i0},\beta_0)(X_{it}-\gamma_i)\mid B_t],
$$

the orthogonalized score $M_t$ is more directly aligned with the first-order slope expansion than the raw score $G_t^\beta$. This is why the publishable version should probably be a **profiled, orthogonalized, time-block stochastic subgradient** paper rather than a raw $(\alpha,\beta)$-SGD paper.

---

# Goal 1: Must-haves

## First publishable draft

The first publishable draft should have the following structure.

### Model

Use Chiang et al.’s model:

$$
(Y_{it},X_{it}')=g(A_i,B_t,U_{it}),
$$

$$
Q_\tau(Y_{it}\mid X_{it},A_i)
=\alpha_0(\tau;A_i)+X_{it}'\beta_0(\tau).
$$

### Objective and scores

Define $Q_{NT}$, $q_t$, $\psi_{it}$, $G_t^\beta$, and the orthogonalized score $M_t$.

### Algorithm

Present one main algorithm:

- profiled/orthogonalized time-block S-subGD; or
- smoothed time-block GD with one-step correction.

Do not present too many algorithms as coequal. Put alternatives in simulations.

### Main theorem

Prove asymptotic equivalence to Chiang et al.’s exact FEQR.

### Inference

Use Chiang et al.’s robust covariance estimator for the first version. Discuss Lee-style random scaling as a second-stage extension.

### Simulations

Compare full FEQR, time-block S-subGD, smoothed time-block GD, and naive observation-wise SGD.

## The proposed first-draft question

The first draft should ask:

> **Can a time-block stochastic subgradient algorithm deliver asymptotically valid, computationally scalable inference for FEQR under pervasive common shocks?**

This question follows directly from the intersection of Chiang et al. (2026) and Lee et al. (2023).

Our key insight is that in a fixed-effects panel quantile regression model with pervasive common shocks, the stochastic unit for online or stochastic-gradient computation should be a **time block** $t$, not an individual observation $(i,t)$.

That insight comes from the dependence structure in [Chiang, Galvao, and Wei (2026)](https://arxiv.org/abs/2602.19201), where the common time shock $B_t$ is shared across all units $i=1,\ldots,N$ within period $t$, while the shocks are independent across $t$. It also comes from their main asymptotic result: the slope estimator is $\sqrt T$-asymptotically normal, not $\sqrt{NT}$-asymptotically normal. Thus, treating the $NT$ cells as if they were the primitive independent online observations would be statistically misleading.

The “time-block” insight should be kept. The full Lee-style random-scaling theorem should not be assumed to transfer automatically, because Lee et al. (2023) study a fixed-dimensional cross-sectional QR recursion, whereas this project has $N$ fixed effects and a common-shock panel dependence structure.

### Why “FEQR under pervasive common shocks” comes from Chiang et al.

Chiang et al. introduce the common-shock panel DGP in Section 2, PDF page 6:

$$
(Y_{it},X_{it}')=g(A_i,B_t,U_{it}).
$$ {#eq-CGW-1}

They specify the fixed-effects quantile model

$$
Q_\tau(Y_{it}\mid X_{it},A_i)
=
\alpha_0(\tau;A_i)+X_{it}'\beta_0(\tau).
$$ {#eq-CGW-2}

After conditioning on $A_i=a_i$, they write

$$
Y_{it}=\alpha_{i0}(\tau)+X_{it}'\beta_0(\tau)+\epsilon_{it},
\qquad
Q_\tau(\epsilon_{it}\mid X_{it},a_i)=0,
$$

and

$$
(X_{it},\epsilon_{it})=g_i(B_t,U_{it}).
$$ {#eq-CGW-3}

This structure makes $B_t$ the source of cross-sectional dependence: at a fixed $t$, every unit shares the same common shock.

### Why “time-block” follows from Chiang et al.’s asymptotics

Chiang et al.’s Theorem 1, PDF page 14, states that under Assumptions 1--6 and

$$
(\log N)^2/T\to0,
$$

$$
\sqrt T(\hat\beta-\beta_0)\Rightarrow N(0,V),
\qquad
V=\Gamma^{-1}\Sigma\Gamma^{-1}.
$$

This is crucial. The effective first-order sampling variation is over $t$, not over all $(i,t)$. Therefore, if one wants an online/stochastic algorithm whose path can be used for inference, its stochastic increments should mimic the $T$-indexed first-order randomness. That means sampling or processing time blocks $t$, not individual cells $(i,t)$.

### Why “computationally scalable” follows from Lee et al.

Lee et al. motivate their paper by the fact that conventional QR inference is computationally difficult for ultra-large samples. On PDF page 1, their abstract says their method uses S-subGD updates, Polyak--Ruppert averaging, and a pivotal statistic computed from the solution path. On PDF page 4, Section 1.2, they focus on scales as large as

$$
(n,d)\sim(10^7,10^3).
$$

On PDF page 10, they define sequential updates

$$
\beta_i=\beta_{i-1}-\gamma_i\nabla q(\beta_{i-1},Y_i),
$$ {#eq-LLSS-8}

$$
\bar\beta_i=\bar\beta_{i-1}\frac{i-1}{i}+\beta_i\frac{1}{i},
$$ {#eq-LLSS-9}

and a pathwise random-scaling matrix $\hat V_i$ in equation (10). Thus, Lee et al. provide the computational architecture that Chiang et al. do not address.

### Why “asymptotically valid inference” is essential

Chiang et al. emphasize that common shocks change both the convergence rate and the covariance structure. Their abstract states that common shocks fundamentally alter the asymptotic covariance and make conventional covariance estimators inconsistent. In Section 4, page 17, Theorem 2 establishes consistency of their robust covariance estimator.

Therefore, a computationally scalable algorithm is not enough. It must preserve the correct Chiang et al. first-order law, or it must provide a new valid inferential law.

### The question is narrow enough to be publishable

The first draft’s question avoids three traps:

1. It does not claim to solve all online inference for panels.
2. It does not claim Lee et al.’s fixed-dimensional path theory transfers automatically to $(N+p)$-dimensional FEQR.
3. It does not add unbalanced panels, MD-QR, and random-scaling all at once.

It instead asks whether the right stochastic unit and the right approximation theorem allow scalable computation while retaining Chiang et al.’s common-shock-valid inference.

## Overview of the theorem spine

The main theorem spine should be:

1. **Algorithm theorem:** the proposed time-block algorithm obtains an approximate FEQR solution with KKT/score residual smaller than $T^{-1/2}$.
2. **Asymptotic equivalence theorem:** the approximate solution is first-order equivalent to the exact Chiang et al. FEQR estimator.
3. **Inference theorem:** Chiang et al.’s robust covariance estimator remains valid for the approximate solution; a stronger Lee-style random-scaling theorem is optional/second-stage.
4. **Negative-control theorem or simulation:** naive observation-wise SGD gives misleading inference under common shocks.

Each piece is explained below.

---

## The algorithm theorem

### Proposed theorem statement

Let $(\tilde\alpha,
\tilde\beta)$ be the output of a time-block stochastic subgradient or smoothed-gradient algorithm after $S$ updates. Let $(\hat\alpha,
\hat\beta)$ be the exact minimizer of $Q_{NT}$. Define a score/KKT residual $R_{NT}(\tilde\alpha,
\tilde\beta)$ measuring violation of the FEQR first-order conditions. A first algorithm theorem would state:

$$
R_{NT}(\tilde\alpha,
\tilde\beta)=o_p(T^{-1/2}).
$$ {#eq-algorithm-residual-target}

Alternatively, if working with objective values:

$$
Q_{NT}(\tilde\alpha,
\tilde\beta)-Q_{NT}(\hat\alpha,
\hat\beta)
=o_p(T^{-1}).
$$

The exact residual metric should be chosen to imply

$$
\|\tilde\beta-
\hat\beta\|=o_p(T^{-1/2}).
$$

### Literature basis

#### Boyd--Mutapcic--Duchi stochastic subgradients

Boyd et al. define a noisy unbiased subgradient $\tilde g$ by requiring

$$
E(\tilde g\mid x)\in\partial f(x).
$$

They consider updates

$$
x^{(k+1)}=x^{(k)}-\alpha_k\tilde g^{(k)}.
$$

On PDF pages 2--3 they discuss convergence under step sizes satisfying square summability but not summability:

$$
\sum_k\alpha_k^2<\infty,
\qquad
\sum_k\alpha_k=\infty.
$$

This is the generic optimization basis for a stochastic subgradient algorithm.

#### Lee et al. stochastic QR update

Lee et al. specialize stochastic subgradients to QR via

$$
\beta_i=\beta_{i-1}-\gamma_i\nabla q(\beta_{i-1},Y_i).
$$

The proposed paper replaces $Y_i$ with the time block $\mathcal Y_t$.

### What is new relative to the literature

Neither Boyd et al. nor Lee et al. directly cover FEQR with $N$ fixed effects and common shocks. The algorithm theorem must therefore show that the time-block stochastic approximation is accurate enough **in the FEQR/common-shock environment**. This can be done either by:

1. proving stochastic-subgradient convergence for the full finite-sample convex objective, then choosing enough iterations that the residual is $o_p(T^{-1/2})$; or
2. using a smoothed objective and showing optimization and smoothing errors are both negligible at the $\sqrt T$ scale; or
3. defining the algorithm as an approximate solver for $Q_{NT}$, and imposing/verifying the residual rate as an algorithmic condition.

The third route is the most defensible first-draft route.

---

## The asymptotic equivalence theorem

### Proposed theorem statement

Let $(\hat\alpha,\hat\beta)$ solve Chiang et al.’s exact FEQR problem. Let $(\tilde\alpha,
\tilde\beta)$ be the algorithmic output. Under Chiang et al.’s statistical assumptions and additional algorithmic accuracy conditions,

$$
\sqrt T(\tilde\beta-
\hat\beta)=o_p(1).
$$ {#eq-AE}

Then, since Chiang et al. prove

$$
\sqrt T(\hat\beta-
\beta_0)\Rightarrow N(0,V),
\qquad
V=\Gamma^{-1}\Sigma\Gamma^{-1},
$$

Slutsky’s theorem gives

$$
\sqrt T(\tilde\beta-
\beta_0)\Rightarrow N(0,V).
$$

### Literature basis

The first-order target law is Chiang et al.’s Theorem 1, PDF page 14. The asymptotic linear representation supporting it is their equation (14), PDF page 27:

$$
\hat\beta-\beta_0
=
\Gamma_N^{-1}
\left(
\tilde H_N^{(2)}(\alpha_0,\beta_0)
-
\frac{1}{N}\sum_{i=1}^N\gamma_i\tilde H_{Ni}^{(1)}(\alpha_{i0},\beta_0)
\right)
+o_p(T^{-1/2}).
$$

The proposed asymptotic-equivalence theorem is a standard econometric strategy: prove that a feasible/computational estimator is closer to the exact estimator than the estimator’s statistical fluctuation. Here the fluctuation is $T^{-1/2}$, not $(NT)^{-1/2}$.

### Why this is easier than proving a new SGD limit

It avoids proving a functional central limit theorem for the whole online path. It only requires that the final output be sufficiently close to the exact FEQR solution. This is a much cleaner first paper because Chiang et al. have already solved the hardest statistical problem: common-shock FEQR asymptotics.

---

## The random-scaling theorem

### First-draft version

The first draft can include a conservative inference theorem:

If

$$
\sqrt T(\tilde\beta-
\hat\beta)=o_p(1)
$$

and $\widehat V$ is Chiang et al.’s robust covariance estimator computed from $(\tilde\alpha,
\tilde\beta)$ rather than $(\hat\alpha,
\hat\beta)$, then

$$
\widehat V(\tilde\alpha,
\tilde\beta)\overset p\to V
$$

and Wald inference based on

$$
T(R\tilde\beta-r)'(R\widehat V R')^{-1}(R\tilde\beta-r)
$$

is valid.

This is not “Lee-style random scaling.” It is Chiang-style robust covariance inference for a fast approximate estimator.

### Second-draft version

The ambitious result would imitate Lee et al.’s Theorem 3.1 with $T$-indexed time-block iterates. A schematic theorem would be:

$$
T(R\bar\beta_T-R\beta_0)'
(R\widehat V_T^{RS}R')^{-1}
(R\bar\beta_T-R\beta_0)
\Rightarrow
W(1)'
\left(\int_0^1\bar W(r)\bar W(r)'dr\right)^{-1}
W(1).
$$

This requires proving a path FCLT for the time-block stochastic approximation. It is a second-stage result because the nuisance $\alpha_i$ and orthogonalization $\gamma_i$ make the recursion far more complex than in Lee et al.

---

## The negative-control theorem or simulation

### Core claim

A naive observation-wise SGD algorithm that samples individual cells $(i,t)$ treats the $NT$ observations as if they were the primitive stochastic units. Under common shocks, this is not aligned with the dependence structure. It may compute a reasonable point estimate, but its pathwise uncertainty will generally be calibrated to the wrong effective sample size.

### Why this follows from Chiang et al.

Chiang et al.’s Theorem 1 gives $\sqrt T$-asymptotics. Their Remark 4 on PDF page 14 explicitly contrasts this with classical FEQR results: common shocks change the effective convergence rate from $\sqrt{NT}$ to $\sqrt T$. Thus, any inferential method whose path treats $(i,t)$ as independent increments risks producing intervals that shrink too quickly.

### What to prove or simulate

A clean first paper can include a simulation rather than a theorem:

- Run full FEQR and compute Chiang-robust confidence intervals.
- Run time-block S-subGD and compute intervals using the same robust covariance or a time-block path statistic.
- Run naive observation-wise SGD and compute Lee-style random scaling as if $n=NT$.
- Show that naive observation-wise SGD undercovers as common-shock strength rises.

This simulation is important because it visually and numerically justifies the paper’s central methodological choice: time blocks, not individual cells.

---

# Goal 2: Should-haves

## Polyak--Ruppert averaged slope recursion

### What Lee et al. do

Lee et al. produce a sequence of stochastic subgradient iterates

$$
\beta_i=\beta_{i-1}-\gamma_i\nabla q(\beta_{i-1},Y_i).
$$

They then average the iterates:

$$
\bar\beta_n=\frac{1}{n}\sum_{i=1}^n\beta_i.
$$

On PDF page 5 they call this the Polyak--Ruppert average. The underlying theoretical ancestry is [Polyak and Juditsky (1992)](https://doi.org/10.1137/0330046), who show that averaging stochastic-approximation trajectories can recover optimal first-order behavior under suitable conditions.

Lee et al.’s operational recursive form appears on PDF page 10, equation (9):

$$
\bar\beta_i
=
\bar\beta_{i-1}\frac{i-1}{i}+\beta_i\frac{1}{i}.
$$

### What “slope recursion” means in this project

In this project, $\beta$ is the common slope and $\alpha_1,\ldots,\alpha_N$ are nuisance fixed effects. A raw full recursion would update the entire vector $(\alpha_1,
\ldots,\alpha_N,\beta)$, but that creates a growing-dimensional stochastic-approximation problem.

A cleaner slope recursion treats $\alpha$ as profiled or auxiliary and updates primarily $\beta$. A generic time-block slope-score recursion would be

$$
\beta_{s+1}
=
\Pi_{\mathcal B}
\left[
\beta_s+
\eta_s\widehat M_{t_s}(\beta_s)
\right],
$$ {#eq-slope-recursion}

where

$$
\widehat M_t(\beta)
\approx
\frac{1}{N}\sum_{i=1}^N
\psi_{it}(\hat\alpha_i(\beta),\beta)(X_{it}-\hat\gamma_i).
$$

Here:

- $t_s$ is the sampled or streamed time block.
- $\hat\alpha_i(\beta)$ is a profiled or running estimate of the fixed effect for unit $i$.
- $\hat\gamma_i$ estimates the orthogonalization term appearing in Chiang et al.’s asymptotic expansion.
- $\Pi_{\mathcal B}$ is projection onto a compact slope parameter space if needed.

Then define the Polyak--Ruppert averaged slope

$$
\bar\beta_S=\frac{1}{S}\sum_{s=1}^S\beta_s.
$$

**Status:** research proposal. It is motivated by Lee et al.’s equations (8)--(9) and Chiang et al.’s projected-score expansion, but it is not proved in either paper.

## Lee-style random-scaling inference

Prove a time-block analog of Lee et al.’s Theorem 3.1 with a profiled/orthogonalized slope recursion.

### What Lee et al. do

Lee et al. construct inference from the path $\{\beta_i\}$. On PDF page 10, equation (10), they define a sequentially computable $\hat V_i$ using accumulated path quantities. Their Theorem 3.1 on PDF page 16 states that for linear restrictions

$$
H_0:R\beta^*=c,
$$

under their assumptions,

$$
n(R\bar\beta_n-c)'(R\hat V_nR')^{-1}(R\bar\beta_n-c)
\Rightarrow
W(1)'
\left(\int_0^1 \bar W(r)\bar W(r)'dr\right)^{-1}
W(1),
$$

where

$$
\bar W(r)=W(r)-rW(1).
$$

For a scalar coefficient, their equation (17) gives the corresponding pivotal $t$-type limit.

The key attraction is that inference uses the solution path rather than estimating the conditional density matrix in the usual QR sandwich formula.

### What “Lee-style” means here

A Lee-style random-scaling theorem in the common-shock FEQR project would use a time-block path

$$
\{\beta_s: s=1,\ldots,S\}
$$

and construct a pathwise scaling matrix analogous to Lee et al.’s $\hat V_n$, but with $S$ corresponding to time-block updates rather than individual observations. A generic version is

$$
\widehat V_S^{RS}
=
S^{-2}
\sum_{s=1}^S
s^2(\bar\beta_s-\bar\beta_S)(\bar\beta_s-\bar\beta_S)'.
$$ {#eq-generic-random-scaling}

The desired second-stage result would be a pivotal limit for

$$
S(R\bar\beta_S-R\beta_0)'
(R\widehat V_S^{RS}R')^{-1}
(R\bar\beta_S-R\beta_0).
$$

If $S=T$, the scaling would have to be compatible with Chiang et al.’s $\sqrt T$-asymptotics.

### Why this is a stronger second draft, not the first draft

A full Lee-style random-scaling theorem is harder than an asymptotic-equivalence theorem for three reasons.

1. **Fixed dimension versus growing nuisance dimension.** Lee et al. analyze a fixed-dimensional parameter $\beta\in\mathbb R^d$. FEQR has $N$ fixed effects plus the slope $\beta$, and $N\to\infty$.
2. **Exact optimizer versus stochastic path.** Chiang et al. prove asymptotic normality for the exact FEQR optimizer. Lee et al. prove pathwise stochastic approximation theory. Combining them requires new pathwise empirical-process control under common shocks.
3. **Nonsmooth objective with nuisance profiling.** The check loss is nonsmooth, and the fixed effects enter through $N$ nuisance intercepts. A random-scaling theorem would need a uniform local linearization of the stochastic recursion after profiling or orthogonalization.

Therefore the first publishable paper should probably prove:

$$
\sqrt T(\tilde\beta-\hat\beta)=o_p(1),
$$

where $\tilde\beta$ is the algorithmic output and $\hat\beta$ is Chiang et al.’s exact FEQR estimator. Then Chiang et al.’s covariance theory applies. A second paper or second-stage section can attempt the full Lee-style path theorem.

---

# Extension 1: [Galvao and Wang (2015)](https://doi.org/10.1016/j.jmva.2014.08.007)

Develop a Galvao--Wang-style minimum-distance estimator whose covariance accounts for common shocks.

## Exact locations for the claims about Galvao and Wang

### Claim: “Their MD-QR estimator is computationally attractive”

**Exact location:** [Galvao and Wang (2015)](https://doi.org/10.1016/j.jmva.2014.08.007), PDF page 2, Introduction.

They state that the MD-QR estimator is “computationally attractive,” “easy to implement in practice, interpret, and replicate,” and that FE-QR can be cumbersome because it estimates all individual intercepts and slopes simultaneously. They explain that MD-QR instead solves regression quantiles for each individual.

A second supporting location is the abstract on PDF page 1, which says the MD-QR estimator is “computationally fast, especially for large cross-sections.”

### Claim: “efficient within a class of minimum-distance estimators”

**Exact locations:**

1. Abstract, PDF page 1: they state that the proposed estimator is “efficient in the class of minimum distance estimators.”
2. Section 2, PDF page 4, after equation (6): they define a class $\mathcal M$ by varying the weight matrices $W_i$ in

$$
\hat\beta
=
\left(\sum_{i=1}^n W_i\right)^{-1}
\sum_{i=1}^n W_i\hat\beta_i.
$$ {#eq-GW-6}

They then say the optimal weights are the inverse asymptotic covariance matrices of the individual slope regression quantiles, and that the estimator in equation (3) is the most efficient estimator among estimators that are linear combinations of the individual regression quantiles.

### Claim: “especially useful for large cross-sections”

**Exact locations:**

1. Abstract, PDF page 1: “computationally fast, especially for large cross-sections.”
2. Introduction, PDF page 2: they say the procedure is “especially economical for large cross-sections” because it solves QR separately for each individual instead of estimating all fixed effects and slopes simultaneously.

### Claim: “they also show FE-QR’s asymptotic linear part can be interpreted as a weighted average of individual QR slope estimators”

**Exact location:** [Galvao and Wang (2015)](https://doi.org/10.1016/j.jmva.2014.08.007), Section 2, PDF page 4.

They write that to show FE-QR belongs to the class of minimum-distance estimators, they show $\hat\beta_{FE}$ is a linear combination of weighted regression quantiles for each individual with $W_i=\Gamma_i$. They use the representation

$$
\hat\beta_{FE}-\beta_0
=
\left(\frac{1}{n}\sum_{i=1}^n\Gamma_i\right)^{-1}
\left\{
\frac{1}{nT}
\sum_{i=1}^n\sum_{t=1}^T
[\tau-1(u_{it}\le0)](x_{it}-\gamma_i)
\right\}
+o_p(1),
$$

then rewrite it as

$$
\hat\beta_{FE}
=
\left(\frac{1}{n}\sum_{i=1}^n\Gamma_i\right)^{-1}
\frac{1}{n}\sum_{i=1}^n\Gamma_i\hat\beta_i+o_p(1).
$$

They conclude that asymptotically $\hat\beta_{FE}$ is a weighted average of the individual QR slope estimators.

### Supporting equations and sections

- Section 2, PDF page 3, equation (1): panel QR model

$$
Q_\tau(y_{it}\mid x_{it},\alpha_{i0}(\tau))
=
\alpha_{i0}(\tau)+x_{it}'\beta_0(\tau).
$$

- Section 2, PDF page 3, equation (2): FE-QR estimator.
- Section 2, PDF page 3, equations (3)--(4): infeasible and feasible MD-QR estimators,

$$
\hat\beta_{MD}
=
\left(\sum_{i=1}^n V_i^{-1}\right)^{-1}
\sum_{i=1}^n V_i^{-1}\hat\beta_i,
$$

and the feasible version with $\hat V_i^{-1}$.

- Section 2, PDF page 4, equations (5)--(6): minimum-distance restrictions and general weighted estimator.
- Section 3, PDF page 6, Theorems 3 and 4: asymptotic normality of MD-QR and FE-QR.

## How Galvao--Wang plots a useful route for this paper

The route is:

$$
\text{time-series QR by individual}
\Rightarrow
\text{minimum-distance aggregation}
\Rightarrow
\text{common-shock robust inference}.
$$

### Step 1: time-series QR by individual

For each unit $i$, estimate an individual QR slope $\hat\beta_i$ using $t=1,
\ldots,T$. This follows Galvao and Wang’s MD-QR construction, where $\hat\beta_i$ is the slope estimator from the individual QR problem.

### Step 2: minimum-distance aggregation

Aggregate the $\hat\beta_i$’s using weights:

$$
\hat\beta_{MD}
=
\left(\sum_i\hat V_i^{-1}\right)^{-1}
\sum_i\hat V_i^{-1}\hat\beta_i.
$$

In Galvao and Wang’s classical setting, inverse covariance weighting is efficient within the minimum-distance class.

### Step 3: common-shock robust inference

To adapt this route to Chiang et al.’s setting, the aggregation and covariance theory would need to account for common shocks. The individual estimates $\hat\beta_i$ are not independent across $i$, because all units share $B_t$. Therefore, the covariance of the aggregated estimator would need to be estimated using time variation and cross-sectional averages, analogous to Chiang et al.’s $\Sigma$ estimator.

A plausible common-shock MD-QR influence object would be a time-indexed aggregate of individual QR scores, something like

$$
\frac{1}{N}\sum_{i=1}^N W_i \varphi_{it},
$$

where $\varphi_{it}$ is the individual QR influence contribution. Then the common-shock robust covariance would be based on the time-series variation of these cross-sectional aggregates.

## Why this should remain an alternative route

The Galvao--Wang route is valuable but should remain alternative for the first draft.

### It changes the estimator

The main Chiang et al. target is the standard unregularized FEQR estimator. A Galvao--Wang route shifts the paper toward MD-QR. That could be a good paper, but it is a different paper.

### It requires large $T$ within each individual

MD-QR estimates a separate time-series QR for each individual. This is natural when $T$ is large enough for individual QR to be reliable. Chiang et al.’s contribution is partly that they allow empirically relevant regimes with $T\ll N$ under $(\log N)^2/T\to0$. If $T$ is only moderately large, individual QR estimates may be noisy or unstable.

### Common shocks violate the independence structure used in Galvao--Wang

Galvao and Wang’s baseline theory assumes independence across individuals. Common shocks deliberately violate cross-sectional independence. Thus, adapting MD-QR to common shocks would require a new covariance and possibly a new efficiency theory.

### It distracts from the Lee-style fast-inference motivation

The central requested synthesis is Chiang et al. plus Lee et al. A Galvao--Wang MD-QR paper might be computationally attractive, especially for large $N$, but it is not the most direct way to import Lee et al.’s online subgradient/random-scaling architecture.

### Best use in the first draft

Use Galvao--Wang as:

1. a literature-motivated alternative estimator;
2. a robustness/simulation comparator;
3. a possible extension section; or
4. a second paper on common-shock robust MD-QR.

---

# Extension 2: [Abrevaya (2013)](https://doi.org/10.1111/j.1368-423X.2012.00389.x)

Use Abrevaya’s missingness-specific logic to define valid unbalanced common-shock FEQR objectives and moments.

## Exact locations for the claims about Abrevaya

### Claim: “His paper is about modifying Chamberlain projections for unbalanced panels”

**Exact locations:**

1. Abstract / Summary, PDF page 1: Abrevaya states that Chamberlain’s projection approach was introduced for balanced panels and that the paper extends it to unbalanced panels.
2. Introduction, PDF page 1: he states that Chamberlain’s original idea of projecting the unobserved fixed effect on covariates from all time periods is not immediately applicable in unbalanced panels, and that he introduces a modified Chamberlain approach in which the fixed-effect projection depends on the form of missingness.
3. Section 2, PDF page 4, equation (2.2): he reproduces the balanced-panel Chamberlain projection

$$
c_i=\psi_0+x_{i1}\lambda_{01}+x_{i2}\lambda_{02}+\cdots+x_{iT}\lambda_{0T}+a_i.
$$ {#eq-A-2.2}

4. Section 3.1, PDF pages 6--7: for $T=3$, he introduces $T_i$-dependent projections. For fully observed units:

$$
c_i=\psi_0+x_{i1}\lambda_{01}+x_{i2}\lambda_{02}+x_{i3}\lambda_{03}+a_i,
$$ {#eq-A-3.2}

and when $t=3$ is missing:

$$
c_i=\psi_0^3+x_{i1}\lambda_{01}^3+x_{i2}\lambda_{02}^3+a_i^3.
$$ {#eq-A-3.3}

5. Section 3.3, PDF page 13, equations (3.25)--(3.28): he gives $s_i$-dependent projections for all missingness configurations:

$$
c_i=\psi_0+x_{i1}\lambda_{01}+x_{i2}\lambda_{02}+x_{i3}\lambda_{03}+a_i
\quad\text{(fully observed)},
$$ {#eq-A-3.25}

$$
c_i=\psi_0^1+x_{i2}\lambda_{02}^1+x_{i3}\lambda_{03}^1+a_i^1
\quad\text{(}t=1\text{ missing)},
$$ {#eq-A-3.26}

$$
c_i=\psi_0^2+x_{i1}\lambda_{01}^2+x_{i3}\lambda_{03}^2+a_i^2
\quad\text{(}t=2\text{ missing)},
$$ {#eq-A-3.27}

$$
c_i=\psi_0^3+x_{i1}\lambda_{01}^3+x_{i2}\lambda_{02}^3+a_i^3
\quad\text{(}t=3\text{ missing)}.
$$ {#eq-A-3.28}

These are the missingness-specific projections referenced earlier.

### Claim: “with missingness-specific projections and GMM moments under strict or sequential exogeneity”

**Exact locations for strict exogeneity:**

- Section 2, PDF page 3, Assumption 2.1:

$$
E(u_i\mid x_i,c_i,s_i)=0.
$$ {#eq-A-Assumption-2.1}

- Section 3.1, PDF pages 7--8, equations (3.6)--(3.7): moment functions implied by the $T_i$-dependent projections under strict exogeneity.
- Section 3.1, PDF page 8, Lemma 3.1: under Assumption 2.1, the moment functions satisfy

$$
E[g(z_i,\theta_0)]=0.
$$

- Section 3.1, PDF page 8, equation (3.8): unweighted GMM estimator minimizes

$$
\left(n^{-1}\sum_{i=1}^n g(z_i,\theta)\right)'
\left(n^{-1}\sum_{i=1}^n g(z_i,\theta)\right).
$$ {#eq-A-3.8}

- Section 3.1, PDF page 9, equation (3.16): weighted GMM objective

$$
\left(n^{-1}\sum_{i=1}^n g(z_i,\theta)\right)'
W
\left(n^{-1}\sum_{i=1}^n g(z_i,\theta)\right).
$$ {#eq-A-3.16}

- Section 3.3, PDF pages 13--14, equations (3.29)--(3.32): moment functions for $s_i$-dependent projections under strict exogeneity.

**Exact locations for sequential exogeneity:**

- Introduction, PDF page 2: Abrevaya emphasizes that the Mundlak approach does not naturally extend when strict exogeneity fails, whereas the modified Chamberlain approach extends naturally to sequential exogeneity.
- Section 3.2, PDF pages 11--12: he defines sequential exogeneity as

$$
E(u_{it}\mid x_{it},x_{i,t-1},\ldots,x_{i1},c_i,s_i)=0.
$$

- Section 3.2, PDF page 11, equations (3.17)--(3.21): only a subset of the original moment conditions remains valid under sequential exogeneity.
- Section 3.2, PDF page 12: he defines $g_{seq}(z_i,\theta)$, the corresponding 2SLS estimator $\hat\theta_{seq,2SLS}$, and the optimal GMM estimator $\tilde\theta_{seq}$.
- Section 3.3, PDF page 14: for $s_i$-dependent projections, he says one drops the necessary orthogonality conditions under sequential exogeneity and defines $g_{seq}$, $\hat\theta_{seq,2SLS}$, and $\tilde\theta_{seq}$ accordingly.

## How Abrevaya is relevant for unbalanced common-shock FEQR

Abrevaya is relevant if the project later allows missing observations in the common-shock FEQR panel.

### The issue in unbalanced common-shock FEQR

A balanced common-shock FEQR objective is

$$
Q_{NT}(\alpha,\beta)=\frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^T
\rho_\tau(Y_{it}-\alpha_i-X_{it}'\beta).
$$

With missing observations, let $s_{it}\in\{0,1\}$ indicate whether $(Y_{it},X_{it})$ is observed. A naive unbalanced objective might be

$$
Q^{unbal}_{NT}(\alpha,\beta)
=
\frac{1}{\sum_{i,t}s_{it}}
\sum_{i=1}^N\sum_{t=1}^T
s_{it}\rho_\tau(Y_{it}-\alpha_i-X_{it}'\beta).
$$

But under common shocks, a time-block algorithm would need time-block objectives such as

$$
q_t^{unbal}(\alpha,\beta)
=
\frac{1}{N_t}\sum_{i=1}^N
s_{it}\rho_\tau(Y_{it}-\alpha_i-X_{it}'\beta),
\qquad
N_t=\sum_{i=1}^N s_{it},
$$

or perhaps a weighting scheme that preserves the target estimand under missingness.

### What Abrevaya contributes conceptually

Abrevaya’s key lesson is that missingness changes the valid projection and moment structure. In his linear fixed-effects setting, the fixed-effect projection must depend on the missingness pattern. Translating this to FEQR suggests that:

1. the fixed-effect/nuisance structure may need to be conditioned on missingness patterns;
2. the valid score moments may differ across missingness configurations;
3. sequential exogeneity or feedback may require dropping moments involving future covariates or future observations;
4. a GMM-style representation could be useful for organizing the valid moments.

### Why this should not be in the first paper

Adding unbalanced panels would introduce another layer of notation, identification, weighting, and missingness assumptions. The first paper already has three hard components: common shocks, FEQR nonsmoothness, and stochastic computation. Abrevaya should therefore be treated as a roadmap for a later extension.

---








# Everything below is miscellaneous


# Theorem-drafting workflow

## Why theorem statements should be drafted first, with assumptions separated

When drafting theorems, ask for theorem statements only, with assumptions separated into:

1. DGP assumptions;
2. quantile smoothness assumptions;
3. common-shock assumptions;
4. algorithmic step-size assumptions;
5. approximation-error assumptions.

### DGP assumptions

We begin with a panel data-generating process:

$$
(Y_{it},X_{it}')=g(A_i,B_t,U_{it}),
$$

with mutual independence of $A_i$, $B_t$, and $U_{it}$. These are statistical assumptions.

### Quantile smoothness assumptions

These include conditional densities at zero, boundedness, differentiability, and full-rank Jacobian conditions. They are needed because QR scores are nonsmooth and the asymptotic expansion depends on local behavior of the conditional distribution near the quantile.

### Common-shock assumptions

These specify the strength and nondegeneracy of the common-shock component. In Chiang et al., $\Sigma$ is defined as the limiting variance of a conditional expectation given $B_1$. These assumptions decide whether the limit is common-shock driven.

### Algorithmic step-size assumptions

These specify $\eta_s$, the number of passes/updates $S$, projection sets, smoothing bandwidths if used, and whether the stochastic gradients are unbiased for the finite-sample objective or only approximately so. These assumptions are not part of Chiang et al.’s statistical theorem, but are needed for a computational theorem.

### Approximation-error assumptions

These state the exact rate at which the algorithmic output approximates the FEQR solution:

$$
\sqrt T(\tilde\beta-
\hat\beta)=o_p(1),
$$

or a sufficient KKT residual/objective-gap condition. This is the bridge between optimization and econometrics.

### Why separation helps

Separating assumptions prevents the paper from hiding statistical restrictions inside algorithmic ones. It also helps referees diagnose which part of the result is doing the work. A panel-data theorist will focus on DGP and common-shock assumptions; a QR theorist will focus on density/smoothness assumptions; a computational econometrician will focus on step sizes and residual rates.

---

# Proof map

## Why the proof map is viable

The suggested proof map is:

$$
\text{Lemma 1: algorithmic residual}
\to
\text{Lemma 2: stochastic equicontinuity}
\to
\text{Lemma 3: projected-score expansion}
\to
\text{Theorem: asymptotic equivalence}.
$$

This is viable because each lemma handles a distinct obstacle.

## Lemma 1: algorithmic residual

**Goal:** show the algorithm output is close enough to satisfying the FEQR first-order/KKT conditions.

A schematic statement:

$$
\|S_{NT}(\tilde\alpha,
\tilde\beta)\|
=o_p(T^{-1/2}),
$$

where $S_{NT}$ is an appropriate subgradient or projected score.

**Why needed:** Chiang et al.’s theorem is for the exact optimizer. A stochastic algorithm may stop before reaching the exact minimizer. The residual controls whether this early stopping is asymptotically negligible.

## Lemma 2: stochastic equicontinuity

**Goal:** show that replacing $(\hat\alpha,
\hat\beta)$ by $(\tilde\alpha,
\tilde\beta)$ changes empirical or projected score terms by a negligible amount.

A schematic target:

$$
\sup_{\|\beta-\beta_0\|\le C T^{-1/2}}
\left\|
S_{NT}(\alpha(\beta),\beta)-S_{NT}(\alpha_0,
\beta_0)-\text{linear term}
\right\|
=o_p(T^{-1/2}).
$$

**Why needed:** Quantile scores are nonsmooth. One needs uniform control in a neighborhood of the true parameter to transfer residual accuracy into parameter accuracy.

## Lemma 3: projected-score expansion

**Goal:** use Chiang et al.’s DGP-induced smoothing to obtain a smooth expansion of the projected score around $(\alpha_0,
\beta_0)$.

Chiang et al.’s Remark 6, PDF pages 15--16, explains the key idea: condition on $B_t$ to obtain projected scores such as

$$
\tilde H_{Ni}^{(1)}(\alpha_i,
\beta)
=
\frac{1}{T}\sum_{t=1}^T
E[\tau-1\{Y_{it}\le \alpha_i+X_{it}'\beta\}\mid B_t],
$$

which are differentiable even though the original score is nonsmooth. This gives Taylor expansions unavailable in classical FEQR proofs.

## Theorem: asymptotic equivalence

Combine:

1. algorithmic residual is negligible;
2. stochastic equicontinuity controls local score differences;
3. projected-score expansion gives local invertibility/Jacobian behavior.

Then obtain

$$
\sqrt T(\tilde\beta-
\hat\beta)=o_p(1).
$$

Finally, apply Chiang et al.’s Theorem 1 and Slutsky.

---

# Simulation design and estimator definitions

## Four estimators/algorithms to compare

The prototype simulation should compare:

1. full FEQR;
2. time-block S-subGD;
3. smoothed time-block GD;
4. naive observation-wise SGD.

Each has a distinct purpose.

---

## Full FEQR

### Definition

Full FEQR solves the exact convex optimization problem

$$
(\hat\alpha,
\hat\beta)
=
\arg\min_{\alpha,
\beta}
\frac{1}{NT}
\sum_{i=1}^N\sum_{t=1}^T
\rho_\tau(Y_{it}-\alpha_i-X_{it}'\beta).
$$

This is the estimator in Chiang et al. equation (4), Galvao--Wang equation (2), and Koenker-type FEQR.

### Prototype implementation

In a simulation, full FEQR can be computed using a linear programming QR solver with unit dummies. For moderate $N,T$, it is the benchmark. For very large $N,T$, it becomes the computationally expensive object that motivates the stochastic algorithm.

### Why needed

Full FEQR is the ground truth estimator for the first paper. If the algorithmic estimator is meant to approximate full FEQR, simulation must show that it matches full FEQR’s bias, RMSE, and coverage.

---

## Time-block S-subGD

### Definition

Time-block stochastic subgradient descent samples or streams an entire time period $t$. At iteration $s$, choose $t_s$ and compute

$$
G_{t_s}^\beta(\alpha_s,
\beta_s)
=
\frac{1}{N}\sum_{i=1}^N
\psi_{it_s}(\alpha_s,
\beta_s)X_{it_s}.
$$

A raw slope update is

$$
\beta_{s+1}=\Pi_{\mathcal B}
\left[\beta_s+
\eta_sG_{t_s}^\beta(\alpha_s,
\beta_s)\right],
$$

because $-G_t^\beta$ is the subgradient of the loss with respect to $\beta$.

A raw fixed-effect update would be

$$
\alpha_{i,s+1}
=
\Pi_{\mathcal A}
\left[\alpha_{i,s}+
\eta_{\alpha,s}\frac{1}{N}\psi_{it_s}(\alpha_s,
\beta_s)\right],
$$

although the scaling and step sizes for $\alpha_i$ require care.

A better publishable algorithm may instead profile $\alpha_i$ or use an orthogonalized slope score.

### Prototype simulation

A prototype can use one of three variants:

1. **Raw full-state time-block S-subGD:** update $\alpha_i$ and $\beta$ jointly.
2. **Profiled time-block S-subGD:** update $\beta$, and periodically set $\alpha_i$ to the empirical $\tau$-quantile of $Y_{it}-X_{it}'\beta$ over $t$.
3. **Orthogonalized time-block S-subGD:** update $\beta$ using $X_{it}-\hat\gamma_i$ rather than $X_{it}$.

The third is closest to Chiang et al.’s expansion.

### Why needed

This is the proposed computational contribution. It respects the common-shock dependence structure because each stochastic increment is a time block.

---

## Smoothed time-block GD

### Definition

Smoothed time-block gradient descent replaces the nonsmooth check loss $\rho_\tau$ with a differentiable approximation $\rho_{\tau,h}$, where $h>0$ is a smoothing bandwidth. For example,

$$
Q_{NT,h}(\alpha,
\beta)
=
\frac{1}{NT}\sum_{i,t}
\rho_{\tau,h}(Y_{it}-\alpha_i-X_{it}'\beta).
$$

A time-block gradient is then

$$
\nabla_\beta q_{t,h}(\alpha,
\beta)
=
-\frac{1}{N}\sum_{i=1}^N
\psi_{it,h}(\alpha,
\beta)X_{it},
$$

where $\psi_{it,h}$ is a smooth approximation to $\tau-1\{r_{it}\le0\}$.

### Difference from time-block S-subGD

- **Time-block S-subGD** uses the exact nonsmooth check loss and a subgradient involving an indicator.
- **Smoothed time-block GD** uses a differentiable approximation and an ordinary gradient.

Smoothing may improve numerical stability and allow easier Taylor expansions, but it introduces smoothing bias and requires bandwidth choice.

### Literature basis

Lee et al. discuss smoothed QR methods, including convolution-type smoothing, as a competing scalable approach. Chiang et al.’s DGP-induced smoothing is different: it is not algorithmic smoothing of the check loss; it arises from conditioning/averaging in the DGP. A smoothed-time-block algorithm would combine algorithmic smoothing with common-shock time-blocking.

### Why needed

Smoothed time-block GD is useful as:

1. a computational comparator;
2. a stability check;
3. a possible proof-friendly algorithm;
4. a way to see whether smoothing bias is small relative to the $T^{-1/2}$ scale.

---

## Naive observation-wise SGD

### Definition

Naive observation-wise SGD samples a single cell $(i_s,t_s)$ and updates using

$$
\psi_{i_s t_s}(\alpha,
\beta)X_{i_s t_s}
$$

rather than the cross-sectional average at time $t_s$. It treats the $NT$ cells as if they were the basic stochastic sequence.

### Why this is wrong under common shocks

At a fixed $t$, all units share $B_t$. Therefore the $(i,t)$ cells are not independent in the way a cross-sectional Lee-style recursion assumes. More importantly, Chiang et al.’s asymptotic rate is $\sqrt T$, not $\sqrt{NT}$. A naive observation-wise random-scaling path will likely understate uncertainty because it sees many more pseudo-independent increments than actually exist.

### Why it is needed in the simulation

It is the negative control. It demonstrates that the paper’s central design choice is not cosmetic. If naive observation-wise SGD has similar point estimates but bad coverage, the simulation will clearly show that the dependence-aware time-block design matters for inference.

---

## Prototype DGP

A simple DGP can follow Chiang et al.’s Monte Carlo structure. Let

$$
Y_{it}=\alpha_i+\beta X_{it}+(1+\gamma X_{it})U_{it},
$$

with common-shock-dependent regressors such as

$$
X_{it}=\lambda B_t+V_{it}+c\alpha_i,
$$

where:

- $B_t$ is a common shock;
- $V_{it}$ and $U_{it}$ are idiosyncratic;
- $\lambda$ controls common-shock strength;
- $\gamma$ controls quantile heterogeneity through scale effects;
- $\alpha_i$ are fixed effects.

The true quantile slope is

$$
\beta(\tau)=\beta+\gamma q_\tau(U),
$$

if $U_{it}$ has $\tau$-quantile $q_\tau(U)$.

---

## Why the Monte Carlo should vary $N$, $T$, shock strength, quantile $\tau$, and step-size schedule

### Vary $N$

Varying $N$ tests the large-cross-section regime and the cross-sectional averaging that estimates common-shock conditional expectations. Chiang et al. allow $T\ll N$, so large $N$ is central.

### Vary $T$

Varying $T$ tests the effective sample size. Since the limit is $\sqrt T$, coverage and RMSE should improve primarily with $T$. If an algorithm’s inference improves as if the sample size were $NT$, that is a warning sign.

### Vary shock strength

Shock strength controls how far the DGP is from classical cross-sectional independence. When shock strength is zero or nearly zero, classical methods may work. As shock strength increases, common-shock robust methods should dominate conventional or naive methods. This dimension directly tests the paper’s reason for existence.

### Vary quantile $\tau$

QR behavior differs across quantiles. Tail quantiles have fewer effective observations near the target, more unstable density estimation, and more difficult optimization. Lee et al. and Chiang et al. both study quantile-specific behavior. A credible simulation should include at least $\tau=0.25,0.5,0.75$, and possibly $0.1,0.9$.

### Vary step-size schedule

Step-size schedules determine stochastic algorithm performance. For example,

$$
\eta_s=\eta_0s^{-a}
$$

with $a\in(1/2,1]$ is standard. Lee et al. recommend a practical rule with $a=0.501$ and a scale-adjusted $\gamma_0$ on PDF page 11, equation (11). Since the algorithm is part of the contribution, the simulation must show whether inference is robust to reasonable step-size choices.

### Are these the only variables?

They are not literally the only variables that matter. Other variables include regressor dimension $p$, fixed-effect distribution, error distribution, bandwidth if smoothing is used, number of passes, initialization, and missingness if unbalanced panels are studied.

They are the variables to **prioritize first** because they align with the paper’s central claims:

- $N,T$: asymptotic regime;
- shock strength: common-shock relevance;
- $\tau$: quantile-specific behavior;
- step size: algorithmic validity.

After the first simulation grid works, add robustness checks for $p$, heavy tails, heteroskedasticity, bandwidth, initialization, and number of passes.