% =====================================================================
%  DESIGN NOTE / METHODS SKELETON  --  Weight-of-Evidence Synaptic
%  Intelligence (WoE-SI)
%
%  Intended use: \input into the TAES paper's methods section, OR compile
%  standalone via the minimal wrapper below. The equations and the
%  positioning prose are pre-written and should be treated as FIXED
%  (verify against the implementation, do not re-derive). Everything the
%  coding agent must supply is marked \TODO{...} or \DESIGNDEC{...}.
%
%  NOTATION COLLISION WARNING (read before touching anything):
%  Denoeux 2019 writes the DS frame of discernment as \Theta and classes
%  as \theta_k. That clashes head-on with the CL convention \theta for
%  network parameters and \Omega_i for SI importance. This note therefore
%  RENAMES the class frame to \mathcal{C}=\{c_1,\dots,c_K\}, keeping
%  \theta for parameters and \Omega for importance. Reconcile the macro
%  names below with the main paper preamble before merging.
% =====================================================================

% ----- minimal standalone wrapper (comment out when \input'd) ---------
% \documentclass[journal]{IEEEtran}
% \usepackage{amsmath,amssymb,bm}
% \begin{document}
% ----------------------------------------------------------------------

% ----- notation macros (RECONCILE with main preamble) -----------------
\providecommand{\TODO}[1]{\textbf{\textcolor{red}{[TODO: #1]}}}
\providecommand{\DESIGNDEC}[1]{\textbf{\textcolor{blue}{[DECISION: #1]}}}
\providecommand{\frame}{\mathcal{C}}                 % class frame of discernment
\providecommand{\param}{\bm{\theta}}                 % network parameter vector
\providecommand{\imp}{\Omega}                         % consolidated importance
\providecommand{\pos}[1]{\left[#1\right]_{+}}        % positive part
\providecommand{\neg}[1]{\left[#1\right]_{-}}        % negative part
\providecommand{\Ic}{\mathcal{I}_2}                   % DS information content (p=2)
% ----------------------------------------------------------------------

\section{Weight-of-Evidence Synaptic Intelligence}
\label{sec:woesi}

% --- Motivation (FIXED prose; align tense/citation keys with the paper) ---
Regularisation-based continual learning protects parameters in proportion
to an estimated importance: the Fisher information in EWC, the path
integral of the loss in Synaptic Intelligence (SI), and the output
sensitivity in MAS. All three summarise importance through a single
scalar that conflates two distinct sources of model confidence: the
\emph{imprecision} of the evidence a parameter supports and the
\emph{conflict} between competing pieces of that evidence. WoE-SI replaces
the scalar tracked by SI with the Dempster--Shafer (DS) information content
of the classifier's output mass function, so that importance reflects each
parameter's accumulated contribution to building \emph{committed} (i.e.,
low-uncertainty, non-vacuous) evidence over a task. The construction follows
the DS reinterpretation of softmax classifiers due to
Denoeux~\cite{denoeux2019logistic} and the path-integral importance of
Zenke et al.~\cite{zenke2017synaptic}. % bib keys to reconcile with the main .bib (Denoeux 2019, arXiv:1807.01846v2; Zenke et al. 2017)

\subsection{DS reinterpretation of the linear readout}
\label{sec:woesi-ds}

Let $\bm{\phi}(\bm{x})\in\mathbb{R}^{J}$ denote the penultimate features
produced by the backbone ($J=512$ for ResNet-18), and let the
classification head be the affine map
\begin{equation}
  z_k \;=\; \sum_{j=1}^{J}\beta_{jk}\,\phi_j(\bm{x}) + \beta_{0k},
  \qquad k = 1,\dots,K,
\end{equation}
with logits $z_k$ over the class frame $\frame=\{c_1,\dots,c_K\}$.
Following Denoeux, each feature contributes a weight of evidence for class
$c_k$ and for its complement,
\begin{equation}
  w_{jk}(\bm{x}) \;=\; \beta_{jk}\,\phi'_j(\bm{x}) + \alpha_{jk},
  \qquad \sum_{j=1}^{J}\alpha_{jk} = \beta_{0k},
  \label{eq:woe}
\end{equation}
where $\phi'_j = \phi_j - \mu_j$ are centred features and $\mu_j$ is a
running estimate of $\mathbb{E}[\phi_j(\bm{x})]$ over the current task.
The offsets $\alpha_{jk}$ are fixed by the Least Commitment Principle.
All reported runs use \texttt{centered\_uniform}: features are centred by the
per-task running mean $\mu_j$ and the offset is the uniform split
$\alpha_{jk}=\beta_{0k}/J$, which satisfies the constraint
$\sum_j\alpha_{jk}=\beta_{0k}$ in \eqref{eq:woe} and is the multi-category
generalisation of Denoeux's binary Least-Commitment solution
(\textsection~4.1). We prefer it to the uncentred variant
(\texttt{raw\_uniform}) because centring removes the arbitrary additive
activation offset, so a weight of evidence reflects a feature's deviation from
its task-mean rather than its absolute scale; this keeps $\Ic(m)$ near zero for
a freshly initialised head and makes the importance comparable across
features. The exact per-class identification of \textsection~4.2
(\texttt{full\_lc}) is not used: it requires solving a constrained
least-commitment program per batch for no observed benefit on this benchmark,
and the implementation deliberately raises \texttt{NotImplementedError} for it.

Splitting each weight of evidence into positive and negative parts gives the
total support for each singleton and its complement,
\begin{equation}
  w_k^{+} = \sum_{j=1}^{J}\pos{w_{jk}},
  \qquad
  w_k^{-} = \sum_{j=1}^{J}\neg{w_{jk}},
  \label{eq:woe-totals}
\end{equation}
with $\pos{u}=\max(0,u)$ and $\neg{u}=\max(0,-u)$. The output mass function
$m$ over $2^{\frame}$ is the orthogonal sum of the resulting simple mass
functions, and the softmax posterior recovered from $m$ via the plausibility
transformation is identical to the ordinary classifier output --- the head
is left unchanged.

The quantity WoE-SI tracks is the DS information content (Least Commitment,
$p=2$), whose non-trivial focal sets are the $K$ singletons $\{c_k\}$ with
weight $w_k^{+}$ and their complements with weight $w_k^{-}$:
\begin{equation}
  \Ic(m) \;=\; \sum_{k=1}^{K}\Big[(w_k^{+})^2 + (w_k^{-})^2\Big].
  \label{eq:I2}
\end{equation}
$\Ic(m)$ is a smooth, non-negative scalar measuring how far the aggregated
evidence is from vacuity; it is small at initialisation and grows as the
model commits to discriminative evidence.\footnote{For $K=2$,
\eqref{eq:I2} coincides with the binary construction of
Denoeux~\textsection 3.1 up to the standard reparameterisation.}
$\Ic(m)$ is exact only at the readout; for a deep backbone its gradient
propagates to all parameters through ordinary backpropagation, so every
$\theta_i$ nonetheless receives an importance (see
Section~\ref{sec:woesi-impl}).

\subsection{Path-integrated evidential importance}
\label{sec:woesi-imp}

WoE-SI is SI with the per-step loss replaced by $\Ic(m)$. Parameter updates
are still driven by the task cross-entropy; $\Ic(m)$ supplies an additional
signal. Over the optimisation trajectory of task $t$, the unnormalised
contribution of parameter $\theta_i$ is the discrete line integral
\begin{equation}
  \omega_i^{(t)} \;=\;
  \sum_{\text{steps}}
  \frac{\partial \Ic(m)}{\partial \theta_i}\,\Delta\theta_i,
  \label{eq:pathint}
\end{equation}
which approximates the share of the total change in committed evidence,
$\Delta\Ic$, attributable to $\theta_i$'s movement. At the end of task $t$
this is normalised by the squared displacement and accumulated across tasks,
\begin{equation}
  \imp_i^{(t)} = \frac{\pos{\omega_i^{(t)}}}
                      {\big(\Delta\theta_i^{(t)}\big)^2 + \xi},
  \qquad
  \imp_i \;\leftarrow\; \imp_i + \imp_i^{(t)},
  \label{eq:consolidate}
\end{equation}
where $\Delta\theta_i^{(t)}$ is the net displacement over the task and $\xi$
is a damping constant. The end-of-task parameters are stored as the anchor
$\theta_i^{\star}$. The rectifier $\pos{\cdot}$ in \eqref{eq:consolidate} is
applied to the path integral $\omega_i^{(t)}$, not to its per-step summands:
importance is granted only to parameters whose \emph{net} movement over the
task increased $\Ic(m)$, i.e.\ that built committed evidence, while parameters
with a net evidence-eroding trajectory are left free. We retain this
convention (the DS analogue of SI protecting loss-reducing parameters): on the
two-task radar split it yielded strictly non-negative backward transfer for
WoE-SI ($\mathrm{BWT}\!=\!-0.014$, versus $-1.0$ for naive fine-tuning), and we
observed no task on which rectification was counterproductive. Removing the
rectifier (signed $\omega_i^{(t)}$) is not recommended: it lets a parameter's
late-task evidence erosion cancel its early-task contribution, under-protecting
parameters that were genuinely important early in the task.

For task $t>1$ the training objective adds the quadratic anchor penalty
\begin{equation}
  \mathcal{L} \;=\; \mathcal{L}_{\mathrm{CE}}
  + \frac{\lambda}{2}\sum_i \imp_i\,
    \big(\theta_i - \theta_i^{\star}\big)^2 ,
  \label{eq:penalty}
\end{equation}
identical in form to SI but with evidential importance. The penalty is
zero on the first task.

\subsection{Implementation and computational notes}
\label{sec:woesi-impl}

WoE-SI operates on the standard linear head; the evidential classifier of
Tong, Xu \& Denoeux~\cite{tong2021evidential} % 2021 evidential head present in the codebase; bib key to reconcile
present in the codebase is not used (it is a valid alternative substrate but is
out of scope here). The
gradient $\partial\Ic(m)/\partial\theta_i$ is obtained by a dedicated
backward pass on \eqref{eq:I2}, kept separate from the cross-entropy
gradient. In the task-incremental setting $\Ic(m)$ is computed over the
task-masked logits; in the class-incremental setting it is computed over the
full shared head. The dedicated $\Ic$ backward uses inference-mode
BatchNorm so it does not perturb the running statistics owned by the
cross-entropy pass, and it is obtained with \texttt{torch.autograd.grad} so it
never overwrites the cross-entropy gradient buffers.

The only cost above SI is this extra backbone forward/backward for $\Ic(m)$.
With per-step exact gradients (\texttt{importance\_stride}$=1$) WoE-SI runs at
$1.34\times$ the per-step wall-clock of SI in our setup
(ResNet1D-18, batch~32, CPU; $59.6$ vs $44.4$\,ms/step). Amortising the $\Ic$
gradient over a window of $k$ steps --- sampling $\partial\Ic/\partial\theta_i$
at the window start and integrating against the window-aggregated displacement
--- reduces this to $1.03\times$ SI at $k=4$. The accuracy/cost trade-off is
mild: $k>1$ assumes the $\Ic$ gradient is roughly constant within the window,
which held on the radar splits without measurable loss in retention; we report
$k=1$ as the default and use $k>1$ only when the importance backward dominates
runtime.

\begin{table}[t]
\caption{WoE-SI hyperparameters and design settings (reported runs).}
\label{tab:woesi-config}
\centering
\begin{tabular}{ll}
\hline
Setting & Value \\
\hline
$\lambda$ (penalty strength)      & $0.4$ (TIL), $0.9$ (CIL) \\
$\xi$ (damping)                   & $10^{-3}$ \\
centering mode                    & \texttt{centered\_uniform} \\
$\mu$ EMA momentum                & $0.9$ \\
importance stride                 & $1$ \\
conflict weighting                & off \\
\hline
\end{tabular}
\end{table}

\subsection{Relationship to existing regularisers}
\label{sec:woesi-related}

% --- FIXED positioning argument; this is the defensibility claim. ---
WoE-SI shares SI's path-integral form \eqref{eq:pathint} and EWC/MAS's
quadratic anchor \eqref{eq:penalty}, but differs in the quantity whose
sensitivity defines importance. Fisher information and SI's loss integral
are agnostic to the structure of the model's uncertainty: a parameter that
drives confident-but-conflicting evidence is indistinguishable from one that
drives confident-and-correct evidence. Because $\Ic(m)$ is a function of the
separated weights of evidence \eqref{eq:woe-totals}, it is sensitive to the
imprecision (vacuity) axis directly, and---under the optional conflict
weighting of Section~\ref{sec:woesi-ablation}---to the conflict axis as
well. Empirically the importance mass concentrates differently from a
loss-driven integral: on the two-task radar split, $\imp$ accumulates
predominantly on the shared backbone rather than the readout (the head being
frozen out of the masked CE path in TIL), and protecting it is what preserves
the earlier task --- WoE-SI retains $98.6\%$ of task-0 accuracy
($\mathrm{BWT}=-0.014$) where naive fine-tuning retains $0\%$
($\mathrm{BWT}=-1.0$), landing in the same regime as the repository's SI/MAS
regularisers. \TODO{For the camera-ready, quantify the parameter-subset
overlap between WoE-SI and SI/MAS (e.g.\ rank-correlation of $\imp$ or
top-$p\%$ Jaccard) on the full benchmark to substantiate that the evidential
importance is not "Fisher under a relabelling"; the \texttt{Net.omega\_summary}
hook exposes the per-group $\imp$ needed for this.}

\subsection{Ablation: conflict-weighted importance}
\label{sec:woesi-ablation}

The default $\Ic(m)$ captures the imprecision/commitment axis only. The
conflict-weighted variant reweights each class term in \eqref{eq:I2} by a
factor derived from the overlap of positive and negative evidence, isolating
the contribution of inter-class conflict to parameter importance. For class
$c_k$ the two simple mass functions place belief $1-e^{-w_k^{+}}$ on
$\{c_k\}$ and $1-e^{-w_k^{-}}$ on its complement; as these focal sets are
disjoint, Dempster's rule assigns the product to the empty set, i.e.\ the
degree of conflict
\begin{equation}
  \kappa_k \;=\; \big(1-e^{-w_k^{+}}\big)\big(1-e^{-w_k^{-}}\big),
  \label{eq:kappa}
\end{equation}
and the conflict-weighted information content reweights each term by
$(1+\kappa_k)$,
\begin{equation}
  \Ic^{\kappa}(m) \;=\; \sum_{k=1}^{K}(1+\kappa_k)
    \Big[(w_k^{+})^2 + (w_k^{-})^2\Big],
  \label{eq:I2kappa}
\end{equation}
so that classes whose evidence is internally conflicting contribute more to
the tracked scalar. The factor satisfies $(1+\kappa_k)\ge 1$, hence
$\Ic^{\kappa}\ge\Ic$ pointwise, and the variant is gated by a single flag
(\texttt{woe\_conflict\_weighting}) that leaves the default $\Ic(m)$ path
untouched. It is disabled in all reported runs.
\TODO{Camera-ready: report whether enabling \eqref{eq:I2kappa} shifts the
selected parameter subset (overlap metric as above) and the resulting BWT/ACC,
to test whether conflict-awareness --- not just imprecision --- carries
continual-learning signal.}

% ----- close standalone wrapper (comment out when \input'd) -----------
% \end{document}
% ----------------------------------------------------------------------