---
identifier: arxiv:2403.00977v1
title: Scaling Up Adaptive Filter Optimizers
authors:
  - Jonah Casebeer
  - Nicholas J. Bryan
  - Paris Smaragdis
published: "2024-03-01T20:58:43+00:00"
url: https://arxiv.org/abs/2403.00977v1
source: arxiv
doi: null
arxiv_id: 2403.00977v1
categories:
  - cs.SD
  - eess.AS
---

# Scaling Up Adaptive Filter Optimizers

Jonah Casebeer    Nicholas J. Bryan    Paris Smaragdis ^(†)^(†)thanks:
J. Casebeer is with the Department of Computer Science, University of
Illinois at Urbana-Champaign, Urbana, IL 61801 USA˜(e-mail:
jonah.casebeer@ieee.org).  
N. J. Bryan is with Adobe Research, San Francisco, CA, 94103
USA˜(e-mail: njb@ieee.org)  
P. Smaragdis is with the Department of Computer Science and Department,
University of Illinois at Urbana-Champaign, Urbana, IL 61801
USA˜(e-mail: paris@illinois.edu)

###### Abstract

We introduce a new online adaptive filtering method called supervised
multi-step adaptive filters (SMS-AF). Our method uses neural networks to
control or optimize linear multi-delay or multi-channel frequency-domain
filters and can flexibly scale-up performance at the cost of increased
compute – a property rarely addressed in the AF literature, but critical
for many applications. To do so, we extend recent work with a set of
improvements including feature pruning, a supervised loss, and multiple
optimization steps per time-frame. These improvements work in a cohesive
manner to unlock scaling. Furthermore, we show how our method relates to
Kalman filtering and meta-adaptive filtering, making it seamlessly
applicable to a diverse set of AF tasks. We evaluate our method on
acoustic echo cancellation (AEC) and multi-channel speech enhancement
tasks and compare against several baselines on standard synthetic and
real-world datasets. Results show our method performance scales with
inference cost and model capacity, yields multi-dB performance gains for
both tasks, and is real-time capable on a single CPU core.

###### Index Terms: 

adaptive filtering, supervised adaptive filtering, acoustic echo
cancellation, beamforming, learning to learn

## I Introduction

Adaptive filters (AF) play an indispensable role in a wide array of
signal processing applications such as acoustic echo cancellation,
equalization, and interference suppression. AFs are parameterized by
time-varying filter weights and require an update or optimization rule
to control them over time. Improving the performance of AFs continues to
pose an intricate challenge, requiring a nuanced approach to optimizer
design. Consequently, AF algorithm designers have relied on mathematical
insights to create tailored optimizers, starting from the foundational
development of the least mean squares algorithm (LMS) \[[1](#bib.bib1)\]
to the Kalman filter \[[2](#bib.bib2), [3](#bib.bib3), [4](#bib.bib4),
[5](#bib.bib5)\].

Fig. 1: Acoustic echo cancellation performance vs. model size,
optimization steps per time-frame (opt. steps), and supervision levels.
Bubble size shows real-time-factor (RTF) where smaller is faster,
inner-shape shows opt. steps, and the vertical dotted line separates
unsupervised (left) and supervised (right) approaches. SMS-AF, in bold
on the far right, demonstrate robust scaling performance in terms of
parameters, and RTF.

In contrast, we have witnessed countless remarkable deep learning
algorithm advancements in other domains through the principle of
“scaling” \[[6](#bib.bib6), [7](#bib.bib7), [8](#bib.bib8),
[9](#bib.bib9)\]. The scaling approach involves improving an existing
method by deploying additional computational resources. Scaling
methodologies are particularly enticing, as they tap into the increasing
computational capabilities of modern smart devices, minimizing the need
for labor-intensive manual tuning and intervention. In the context of
neural networks for online low-latency AFs that can benefit from
scaling, we find two general approaches: 1) model-based methods that
integrate deep neural networks (DNNs) into existing AF frameworks to
update optimizer statistics \[[10](#bib.bib10), [11](#bib.bib11),
[12](#bib.bib12)\], step-size \[[13](#bib.bib13), [14](#bib.bib14)\], or
other quantities \[[15](#bib.bib15)\] and 2) model-free strategies that
do not rely on an existing AF strategy, learn AF optimizers using
meta-learning in an end-to-end fashion \[[16](#bib.bib16),
[17](#bib.bib17), [18](#bib.bib18), [19](#bib.bib19)\], and yield
state-of-the-art (SOTA) results \[[15](#bib.bib15), [19](#bib.bib19)\].
We focus on the latter, given their recent success, but note scaling
such approaches have either been limited to high-latency
regimes \[[18](#bib.bib18)\] or only marginally improves
results \[[15](#bib.bib15)\].

We propose a new online AF method called supervised multi-step AF
(SMS-AF). Our method integrates a series of algorithm improvements on
top of recent meta-learning methods \[[17](#bib.bib17),
[18](#bib.bib18)\] that together enable scaling performance by
increasing model capacity and/or inference cost as shown
in Fig. [1](#S1.F1 "Fig. 1 ‣ I Introduction ‣ Scaling Up Adaptive Filter Optimizers").
We evaluate our approach on the tasks of acoustic echo
cancellation (AEC) and generalized sidelobe canceller (GSC) speech
enhancement and compare to recent SOTA approaches. Results show that our
scaling behavior translates to substantial performance gains in all
metrics across tasks and datasets and delivers breakthrough AEC
performance

Our contributions include: 1) A new general purpose AF method that allow
us to reliably improve performance by simply using more computation, 2)
Design insights for customizing our proposed method for the task of AEC
and GSC, 3) Empirical exploration of AF optimizer scaling showing our
approach scales vs. model size and optimizer step count, and 4) Insights
as to how our approach generalizes Kalman filtering.

## II Background

### II-A Adaptive Filters

An AF is an optimization procedure that seeks to adapt filter parameters
to fit an objective over time. AFs typically input a mixture
$`\underline{\mathbf{d}}[\tau]`$, adjust a time-varying linear filter
$`h_{\bm{\theta}[\tau]}`$ with parameters $`\bm{\theta}[\tau]`$ to
remove noise via knowledge from a reference signal
$`\underline{\mathbf{u}}[\tau]`$, produce estimate
$`\underline{\mathbf{y}}[\tau]`$, and output an error signal
$`\underline{\mathbf{e}}[\tau]`$. We focus on multi-delay and/or
multi-channel frequency-domain filters (MDF) for low-latency processing.
The filter parameters are updated across time by minimizing a loss,
$`\mathcal{L}(h_{\bm{\theta}[\tau-1]},\cdots)`$, resulting in a per
frame $`\tau`$ update rule,

|     |     |     |     |
| --- | --- | --- | --- |
|     |

       ``` math
       \bm{\theta}[\tau]=\bm{\theta}[\tau-1]+\bm{\Delta}[\tau].
       ```                                                       |     | (1) |

$`\bm{\Delta}[\tau]`$ can also be written as the output of an optimizer
$`g_{\bm{\phi}}(\bm{\xi}[\tau])=\bm{\Delta}[\tau]`$ with input
$`\bm{\xi}[\tau]`$, and parameters $`\bm{\phi}`$.

### II-B Adaptive Filter Optimizers

The AF optimizer, $`g_{\bm{\phi}}(\bm{\xi}[\tau])`$ is key and can vary
in levels of sophistication. In the simple case, the optimizer can be a
hand-derived algorithm such as LMS. In this case, each parameter in
$`\bm{\theta}`$ is updated independently, so $`g`$ accepts the gradient
with respect to the loss via $`\bm{\xi}[\tau]`$, and is only
parameterized by the step-size $`\bm{\phi}=\{\lambda\}`$, resulting in
$`g`$ simply scaling the gradient. Most AFs operate via the following
steps \[[4](#bib.bib4)\]:

|     |                                               |                    |                                                                                                      |     |     |
| --- | --------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\underline{\mathbf{e}}[\tau]`$ | $`\displaystyle=`$ | $`\displaystyle h_{\bm{\theta}[\tau-1]}(\underline{\mathbf{d}}[\tau],\underline{\mathbf{u}}[\tau])`$ |     | (2) |
|     | $`\displaystyle\bm{\Delta}[\tau]`$            | $`\displaystyle=`$ | $`\displaystyle g_{\bm{\phi}}(\bm{\xi}[\tau],\cdots)`$                                               |     | (3) |
|     | $`\displaystyle\bm{\theta}[\tau]`$            | $`\displaystyle=`$ | $`\displaystyle\bm{\theta}[\tau-1]+\bm{\Delta}[\tau],`$                                              |     | (4) |

where ([2](#S2.E2 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))
applies the
filter, ([3](#S2.E3 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))
updates the optimizer,
and ([4](#S2.E4 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))
updates the filter parameters for the next frame. Optimizers typically
use filter output $`\underline{\mathbf{e}}[\tau]`$, produced via
$`\bm{\theta}[\tau-1]`$, creating a feedback loop. The Kalman filter
(KF) extends the above via distinct “predict” and “update” steps. In the
KF predict step,
([2](#S2.E2 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))-([4](#S2.E4 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))
are run as normal. In the KF update step, however, the filter output is
reprocessed after
([4](#S2.E4 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))
using the latest data:

|     |                                               |                    |                                                                                                     |     |     |
| --- | --------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\underline{\mathbf{e}}[\tau]`$ | $`\displaystyle=`$ | $`\displaystyle h_{\bm{\theta}[\tau]}(\underline{\mathbf{d}}[\tau],\underline{\mathbf{u}}[\tau]).`$ |     | (5) |

### II-C Learned Optimizers

Historically, AFs are hand-derived, given a loss and filter. In
contrast, neural-AF optimizers can be trained via meta-learning
(Meta-AF) \[[20](#bib.bib20), [16](#bib.bib16), [17](#bib.bib17)\].
Meta-AFs are trained to control MDF filters via recurrent neural
networks (RNNs) with parameters $`\bm{\phi}`$ that are trained to
maximize AF performance on a large dataset via an unsupervised (or
self-supervised) meta-objective
$`\mathcal{L}_{M}(\;g_{\bm{\phi}},\mathcal{L}(h_{\bm{\theta}},\cdots)\;)`$
and backpropagation through time (BPTT). A common meta-loss is,

|     |                                                                |                    |                                                                     |     |     |
| --- | -------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\mathcal{L}_{M}(\underline{\bar{\mathbf{e}}})`$ | $`\displaystyle=`$ | $`\displaystyle\ln E[\|\underline{\bar{\mathbf{e}}}[\tau]\|^{2}],`$ |     | (6) |

where
$`\underline{\bar{\mathbf{e}}}[\tau]=\mathrm{cat}(\underline{\mathbf{e}}[\tau],\cdots,\underline{\mathbf{e}}[\tau+L-1])\in\mathbb{R}^{RL}`$,
$`L`$ is the truncation length, $`R`$ is the hop size, and
$`\mathrm{cat}`$ concatenates.

Two important extensions to Meta-AF include 1) higher-order Meta-AF
(HO-Meta-AF) \[[18](#bib.bib18)\], which introduces learnable coupling
modules to model groups of filter parameters, reduce complexity, and
improve performance for high-latency single-block frequency-domain
filters and 2) low-complexity neural Kalman filtering
(NKF) \[[15](#bib.bib15)\], which extends Meta-AF with a KF, a
supervised loss, and different training setup. We regard Meta-AF as SOTA
for unsupervised AFs (see Table IV \[[17](#bib.bib17)\]) and NKF as SOTA
for supervised AFs \[[15](#bib.bib15)\].

## III Scaling Up Learned Optimizers

As the foundation of our SMS-AF method, we combine
Meta-AFs \[[17](#bib.bib17)\] with a higher-order
optimizer \[[18](#bib.bib18)\] with per-frequency inputs
$`\bm{\xi}_{\mathrm{k}}[\tau]`$, and then extend it with three
task-agnostic improvements and one task-specific change. Our training
and inference methods are summarized in
Alg. [1](#alg1 "Algorithm 1 ‣ III-C Scaling Up Feedback: Multi-Step Optimization ‣ III Scaling Up Learned Optimizers ‣ Scaling Up Adaptive Filter Optimizers").

### III-A Scaling Up Feature Quality: Feature Pruning

Our first insight is to use only three features to control filter
adaptation: knowledge of the filter input, final filter output, and
filter state. Compared to past work \[[17](#bib.bib17)\] that uses,

|     |                                                                                                                                                                                                   |     |     |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\bm{\xi}_{\mathrm{k}}[\tau]=[\nabla_{\mathrm{k}}[\tau],\mathbf{u}_{\mathrm{k}}[\tau],\mathbf{d}_{\mathrm{k}}[\tau],\mathbf{e}_{\mathrm{k}}[\tau],\mathbf{y}_{\mathrm{k}}[\tau]],`$ |     | (7) |

where $`\nabla_{\mathrm{k}}[\tau]`$ is a gradient w.r.t. loss
$`\mathcal{L}(\cdots)`$, we use

|     |                                                                                                                                            |     |     |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------ | --- | --- |
|     | $`\displaystyle\bm{\xi}_{\mathrm{k}}[\tau]=[\mathbf{u}_{\mathrm{k}}[\tau],\mathbf{e}_{\mathrm{k}}[\tau],\bm{\theta}_{\mathrm{k}}[\tau]].`$ |     | (8) |

Pruning reduces complexity by lowering input dimension and memory
requirements for the optimizer, while eliminating inference-time
gradients, as shown in line $`8`$ of
Alg. [1](#alg1 "Algorithm 1 ‣ III-C Scaling Up Feedback: Multi-Step Optimization ‣ III Scaling Up Learned Optimizers ‣ Scaling Up Adaptive Filter Optimizers").

### III-B Scaling Up Supervision: Supervised Loss

Our second insight is to use a high-quality supervised loss, instead of
an unsupervised loss. Previous methods have explored supervised losses
such as frame-wise independent supervised losses for
echoes \[[15](#bib.bib15)\] or oracle filter
parameters \[[11](#bib.bib11)\]. These methods treat frequency bins,
adjacent frames, and other channels as distinct optimization entities
and have not scaled \[[15](#bib.bib15)\]. As such, we compute our
supervised loss in the time-domain after all AF operations have been
performed. This strategy is similar
to ([6](#S2.E6 "In II-C Learned Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers")),
but with supervision. Our supervision is non-causal; the loss at
$`\tau`$ depends on updates from $`<\tau`$, enabling the optimizer to
learn anticipatory updates. For AEC,

|     |                                                                      |                    |                                                                                                                     |     |     |
| --- | -------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\mathcal{L}_{S}(\mathbf{d}_{\mathbf{u}},\mathbf{e})`$ | $`\displaystyle=`$ | $`\displaystyle\ln E[\|\underline{\bar{\mathbf{d}}}_{\mathbf{u}}[\tau]-\underline{\bar{\mathbf{e}}}[\tau]\|^{2}],`$ |     | (9) |

where
$`\mathbf{d}_{\mathbf{u}}[\tau]=\mathbf{u}[\tau]\ast\mathbf{w}[\tau]`$
is the true echo. For GSC, we use scale-invariant signal-to-distortion
ratio (SI-SDR). Better loss functions exclusively impact the training
phase, without contributing to test-time complexity, making this change
cost-free for inference. This corresponds to line $`22`$ of
Alg. [1](#alg1 "Algorithm 1 ‣ III-C Scaling Up Feedback: Multi-Step Optimization ‣ III Scaling Up Learned Optimizers ‣ Scaling Up Adaptive Filter Optimizers").

### III-C Scaling Up Feedback: Multi-Step Optimization

Our third insight is to leverage the iterative nature of optimizers by
executing multiple optimization steps per time frame. By doing so, we
offer our optimizers a more powerful feedback mechanism and use the most
current parameters for the filter output. Specifically, we run our
optimizer update via
([2](#S2.E2 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))-([4](#S2.E4 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers")),
([2](#S2.E2 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))-([5](#S2.E5 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers")),
or looping over
([2](#S2.E2 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))-([5](#S2.E5 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))
multiple times. The first option follows Meta-AF, the second option
follows a typical KF, and the third extends the KF. We denote the number
of
([2](#S2.E2 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))-([4](#S2.E4 "In II-B Adaptive Filter Optimizers ‣ II Background ‣ Scaling Up Adaptive Filter Optimizers"))
iterations via $`C`$. Incorporating multi-step optimization in
Alg. [1](#alg1 "Algorithm 1 ‣ III-C Scaling Up Feedback: Multi-Step Optimization ‣ III Scaling Up Learned Optimizers ‣ Scaling Up Adaptive Filter Optimizers")
involves three changes. First, initializing each frame’s filter and
optimizer state with results from the last frame (lines $`3-4`$).
Second, iteratively progressing through steps withing a frame (line
$`5`$), while updating filter parameters/outputs, and optimizer
state (lines $`8-10`$). Last, running a final filter forward pass using
the latest parameters (line $`11`$). We find this approach to be a
compelling alternative to increasing the dimension of the optimizer,
$`H`$. Notably, it avoids increasing the parameter count, and it
linearly scales complexity, in stark contrast to the quadratic
complexity effects associated with $`H`$.

1: function
Forward($`g_{\bm{\phi}},h_{\bm{\theta}},\underline{\mathbf{u}},\underline{\mathbf{d}}_{\mathrm{m}}`$)

2:   for $`\tau\leftarrow 0\textrm{ to }L`$ do

3:    $`\bm{\theta}_{\mathrm{k}}[\tau]`$ $`\leftarrow`$
$`\bm{\theta}_{\mathrm{k}}[\tau-1]`$ $`\triangleright`$ Initialize with
last estimate

4:    $`\bm{\psi}_{\mathrm{k}}[\tau]`$ $`\leftarrow`$
$`\bm{\psi}_{\mathrm{k}}[\tau-1]`$ $`\triangleright`$ Initialize with
last state

5:    for $`\mathrm{c}\leftarrow 0\textrm{ to }C`$ do $`\triangleright`$
For each PU iteration

6:      $`\underline{\mathbf{e}}[\tau]`$ $`\leftarrow`$
$`h_{\bm{\theta}[\tau]}(\underline{\mathbf{d}}[\tau],\underline{\mathbf{u}}[\tau])`$

7:      for $`\mathrm{k}\leftarrow 0\textrm{ to }K`$ do
$`\triangleright`$ Predict step

8:       $`\bm{\xi}_{\mathrm{k}}[\tau]`$ $`\leftarrow`$
$`[\mathbf{u}_{\mathrm{k}}[\tau],\mathbf{e}_{\mathrm{k}}[\tau],\bm{\theta}_{\mathrm{k}}[\tau]`$

9:
      $`(\bm{\Delta}_{\mathrm{k}}[\tau],\bm{\psi}_{\mathrm{k}}[\tau])`$
$`\leftarrow`$
$`g_{\bm{\phi}}(\bm{\xi}_{\mathrm{k}}[\tau],\bm{\psi}_{\mathrm{k}}[\tau])`$

10:       $`\bm{\theta}_{\mathrm{k}}[\tau]`$ $`\leftarrow`$
$`\bm{\theta}_{\mathrm{k}}[\tau]+\bm{\Delta}_{\mathrm{k}}[\tau]`$

        

11:    $`\underline{\mathbf{e}}[\tau]`$ $`\leftarrow`$
$`h_{\bm{\theta}[\tau]}(\underline{\mathbf{d}}[\tau],\underline{\mathbf{u}}[\tau])`$
$`\triangleright`$ Update step   

12:   $`\bar{\mathbf{e}}`$ $`\leftarrow`$
Cat$`(\underline{\mathbf{e}}[\tau]\;\forall\tau)`$

13:   return $`\bar{\mathbf{e}},\bm{\psi}[\tau],h_{\bm{\theta}[\tau]}`$

14: function Train($`\mathcal{D}`$)

15:   $`\bm{\phi}`$ $`\leftarrow`$ \[[21](#bib.bib21)\] init

16:   while $`\bm{\phi}`$ not Converged do

17:    $`\underline{\mathbf{u}},\underline{\mathbf{d}}_{\mathrm{m}}`$
$`\leftarrow`$ Sample($`\mathcal{D}`$) $`\triangleright`$ Get batch from
$`\mathcal{D}`$

18:    $`\bm{\theta},\bm{\psi}`$ $`\leftarrow`$
$`\mathbf{0},\mathbf{0}`$

19:    for $`n\leftarrow 0\textrm{ to end }`$ do

20:      $`\underline{\mathbf{u}},\underline{\mathbf{d}}_{\mathrm{m}}`$
$`\leftarrow`$
NextL$`(\underline{\mathbf{u}},\underline{\mathbf{d}}_{\mathrm{m}})`$
$`\triangleright`$ Grab $`L`$ frames

21:      $`\underline{\mathbf{u}},\bm{\psi},h_{\bm{\theta}}`$
$`\leftarrow`$
Forward($`g_{\bm{\phi}},\bm{\psi},h_{\bm{\theta}},\underline{\mathbf{u}},\underline{\mathbf{d}}_{\mathrm{m}}`$)

22:      $`L_{S}`$ $`\leftarrow`$ $`\mathcal{L}_{S}(\cdots)`$
$`\triangleright`$ Task-dependent objective

23:      $`\bm{\nabla}`$ $`\leftarrow`$ Grad($`L_{S}`$, $`\bm{\phi}`$)
$`\triangleright`$ Truncated BPTT

24:      $`\bm{\phi}[n+1]`$ $`\leftarrow`$
MetaOpt($`\bm{\phi}[n],\bm{\nabla}`$) $`\triangleright`$ i.e. Adam
     return $`\hat{\bm{\phi}}`$

Algorithm 1 Training and inference algorithm.

### III-D Modifications to Overlap-Save for AEC

When using overlap-save, we noticed artifacts due to rapid filter
adaptation. So, we applied a straightforward solution: overlap-add with
a synthesis window, but no analysis window.

### III-E Perspectives

Our modifications are a notable departure from the Meta-AF methodology,
but still aim to learn a neural optimizer for AFs end-to-end. First, by
pruning inputs features and introducing a supervised loss, we eliminate
the need for explicit meta-learning, leading to a more streamlined BPTT
training process. Second, we replace the past unsupervised loss with a
new, strong supervised signal and loss, helping us scale up. Third, we
leverage a multi-step optimization scheme. This creates a generalization
of the Kalman filter, where all parameters are entirely learned, while
retaining explicit predict and update steps. This also effectively
deepens our optimizer networks by sharing parameters across layers in a
depth-wise manner.

## IV Experimental Design

### IV-A Experiments

The goal of our experiments is to benchmark SMS-AF and demonstrate how
it scales. To do so, we perform an initial within method ablation, then
study scaling on AEC and GSC tasks and vary 1) optimizer model sizes
with small (S), medium (M), and large (L) models 2) an unsupervised (U)
or supervised (S) loss and 3) the number of predict (P) and update (U)
steps per frame. Each experiment is labeled with an identifier (e.g.
S$`\cdot`$S$`\cdot`$PU), indicating the size, supervision, and number of
PU steps. We label baselines when applicable.

### IV-B AEC Configuration

Our AEC signal model is
$`\underline{\mathbf{d}}[{\mathrm{t}}]=\underline{\mathbf{u}}[{\mathrm{t}}]\ast\underline{\mathbf{w}}+\underline{\mathbf{n}}[{\mathrm{t}}]+\underline{\mathbf{s}}[{\mathrm{t}}]`$,
where $`\underline{\mathbf{n}}`$ stands for noise, and
$`\underline{\mathbf{s}}`$ is speech. The goal is to recover the speech
$`\underline{\mathbf{s}}`$ given the far-end $`\underline{\mathbf{u}}`$,
and mixture $`\underline{\mathbf{d}}`$. This involves fitting a filter
to mimic $`\underline{\mathbf{w}}`$. We use a linear MDF filter with
$`8`$ blocks, each of size $`512`$, a hop of $`256`$, and construct the
output using overlap-add with a Hann window. Our baselines are NLMS,
KF \[[22](#bib.bib22)\], Meta-AF \[[17](#bib.bib17)\],
HO-Meta-AF \[[18](#bib.bib18)\], and Neural-Kalman
Filter \[[15](#bib.bib15)\]. We also test several HO-Meta-AF model sizes
as well as multi-step NLMS, KF, and HO-Meta-AF. For training, we use the
synthetic fold of the Microsoft AEC Challenge \[[23](#bib.bib23)\]. Each
scene has double-talk, near-end noise and loud-speaker nonlinearities.
We also evaluate on the real, crowd-sourced, blind
test-set \[[23](#bib.bib23)\]. We use echo return loss
enhancement (ERLE) \[[24](#bib.bib24)\] to measure echo reduction. To
describe perceptual quality, we use AEC-MOS, a reference-free model that
predicts a 5 point score \[[23](#bib.bib23)\]. On real data, we prefix
with an R, use ERLE in single-talk and AEC-MOS in double-talk. To
quantify complexity, we use mega FLOP (MFLOP) counts, single core
real-time-factor (RTF) equal to processing over elapsed time, and model
size.

### IV-C GSC Configuration

For GSC, we use a single-block frequency-domain GSC beamformer. The
signal model at each of the $`M`$ microphones is
$`\underline{\mathbf{u}}_{{\mathrm{m}}}[{\mathrm{t}}]=\underline{\mathbf{r}}_{{\mathrm{m}}}[{\mathrm{t}}]\ast\underline{\mathbf{s}}[{\mathrm{t}}]+\underline{\mathbf{n}}_{{\mathrm{m}}}[{\mathrm{t}}]`$,
and $`\underline{\mathbf{r}}_{{\mathrm{m}}}[{\mathrm{t}}]`$ is the
impulse response from source to mic $`{\mathrm{m}}`$. The goal is to
recover the clean speech $`\underline{\mathbf{s}}`$ given the input
signal $`\underline{\mathbf{u}}_{{\mathrm{m}}}[{\mathrm{t}}]`$. This
requires fitting a filter to remove the effects of noise,
$`\underline{\mathbf{n}}_{{\mathrm{m}}}`$. We assume access to a
steering vector and compare against NLMS, recursive-lease-squares (RLS),
and Meta-AF. We test multiple HO-Meta-AF model sizes as well as
multi-step NLMS, RLS, and HO-Meta-AF. We used the
CHIME-3 \[[25](#bib.bib25)\] dataset. For overall quality, we compute
scale-invariant signal-to-distortion
ratio (SI-SDR) \[[26](#bib.bib26)\], and contrast signal-to-interference
ratio (SIR), and signal-to-artifact ratio (SAR) \[[27](#bib.bib27)\].
For perceptual quality, we use Short-Time Objective Intelligibility
(STOI) \[[28](#bib.bib28)\].

### IV-D Optimizer Configuration

For AEC and GSC, we use higher-order Meta-AF optimizers with banded
coupling, and a group size of $`5`$ \[[18](#bib.bib18)\]. This amounts
to a Conv1D layer, two GRU layers, and a transposed Conv1D layer. To
train, we use Adam with a batch of $`16`$, a learning rate of
$`10^{-4}`$, and randomize the truncation length $`L`$ with a maximum of
$`128`$. We apply gradient clipping and reduce the learning rate by half
if the validation performance does not improve for $`10`$ epochs, and
stop training after $`30`$ epochs with no improvement. We use log-MSE
loss on the echo for AEC, and SI-SDR loss on the clean speech for GSC.
All models are trained on one GPU. Note, our S, M, and L model sizes
correspond to $`g_{\bm{\phi}}`$ hidden state sizes of $`16`$, $`32`$,
and $`64`$ with parameters counts of about $`5`$K, $`16`$K, and $`57`$K.

## V Results & Discussion

|                                            |     |                  |     |                    |     |                      |     |                        |     |                      |     |                   |
| ------------------------------------------ | --- | ---------------- | --- | ------------------ | --- | -------------------- | --- | ---------------------- | --- | -------------------- | --- | ----------------- |
| Model                                      |     | ERLE$`\uparrow`$ |     | R-ERLE$`\uparrow`$ |     | AEC- MOS$`\uparrow`$ |     | R-AEC- MOS$`\uparrow`$ |     | MFLOPs$`\downarrow`$ |     | RTF$`\downarrow`$ |
| Mixture                                    |     | \-               |     | \-                 |     | 2.33                 |     | 2.69                   |     | \-                   |     | \-                |
| NLMS$`\cdot`$P                             |     | 4.37             |     | 1.74               |     | 3.16                 |     | 3.06                   |     | 0.08                 |     | 0.31              |
| KF$`\cdot`$P                               |     | 5.44             |     | 2.71               |     | 3.32                 |     | 3.18                   |     | 0.12                 |     | 0.32              |
| KF$`\cdot`$PU                              |     | 6.65             |     | 3.83               |     | 3.65                 |     | 3.38                   |     | 0.12                 |     | 0.35              |
| KF$`\cdot`$PUx2                            |     | 7.15             |     | 5.58               |     | 3.89                 |     | 3.66                   |     | 0.18                 |     | 0.39              |
| M$`\cdot`$U$`\cdot`$P \[[17](#bib.bib17)\] |     | 6.30             |     | 2.99               |     | 3.62                 |     | 3.31                   |     | 9.02                 |     | 0.41              |
| S$`\cdot`$U$`\cdot`$P                      |     | 7.09             |     | 3.30               |     | 3.57                 |     | 3.26                   |     | 2.80                 |     | 0.36              |
| M$`\cdot`$U$`\cdot`$P                      |     | 7.85             |     | 3.75               |     | 3.62                 |     | 3.30                   |     | 7.07                 |     | 0.39              |
| L$`\cdot`$U$`\cdot`$P                      |     | 8.03             |     | 3.91               |     | 3.65                 |     | 3.33                   |     | 20.42                |     | 0.48              |
| NKF$`\cdot`$S$`\cdot`$PU                   |     | 9.29             |     | 6.38               |     | 3.69                 |     | 3.59                   |     | 10.16                |     | \-                |
| S$`\cdot`$S$`\cdot`$P                      |     | 8.63             |     | 3.81               |     | 3.63                 |     | 3.31                   |     | 2.80                 |     | 0.36              |
| M$`\cdot`$S$`\cdot`$P                      |     | 9.84             |     | 4.62               |     | 3.73                 |     | 3.40                   |     | 7.07                 |     | 0.39              |
| L$`\cdot`$S$`\cdot`$P                      |     | 11.62            |     | 5.52               |     | 3.83                 |     | 3.50                   |     | 20.42                |     | 0.48              |
| S$`\cdot`$S$`\cdot`$PU                     |     | 9.13             |     | 6.05               |     | 3.85                 |     | 3.55                   |     | 2.81                 |     | 0.38              |
| M$`\cdot`$S$`\cdot`$PU                     |     | 11.22            |     | 7.87               |     | 3.99                 |     | 3.72                   |     | 7.08                 |     | 0.41              |
| L$`\cdot`$S$`\cdot`$PU                     |     | 13.34            |     | 9.78               |     | 4.05                 |     | 3.85                   |     | 20.43                |     | 0.49              |
| S$`\cdot`$S$`\cdot`$PUx2                   |     | 9.80             |     | 6.96               |     | 3.85                 |     | 3.66                   |     | 5.56                 |     | 0.50              |
| M$`\cdot`$S$`\cdot`$PUx2                   |     | 11.77            |     | 8.86               |     | 4.04                 |     | 3.80                   |     | 14.11                |     | 0.56              |
| L$`\cdot`$S$`\cdot`$PUx2                   |     | 14.25            |     | 11.15              |     | 4.12                 |     | 3.94                   |     | 40.81                |     | 0.69              |

TABLE I: AEC performance vs. computational cost.

### V-A Initial Within Method Ablation

We perform an initial within-method ablation on the task of AEC to
understand our modifications. First, we compare an M$`\cdot`$U$`\cdot`$P
variant trained with the full feature vs. pruned feature set. The full
feature set achieves an ERLE of $`6.39`$dB (not shown), while the pruned
set scores $`7.85`$dB, over a dB gain. We expand on the pruned model and
add supervision, resulting in an ERLE improvement to $`9.84`$dB, a gain
of nearly $`2`$dB. We then use multiple steps per frame. Extending the
supervised model with an update-step increases ERLE to $`11.22`$dB, and
doubling the iterations reaches $`11.77`$dB. To mitigate clicking
artifacts, we then add our modified OLA scheme. This change reduces the
ERLE by $`\approx 1`$ dB ERLE, but removes severe clicking artifacts.
Combined, this yields a $`4`$dB ERLE gain.

### V-B AEC Scaling Ablation and Benchmarking

Next, we explore scaling in AEC as shown
in Fig. [1](#S1.F1 "Fig. 1 ‣ I Introduction ‣ Scaling Up Adaptive Filter Optimizers")
and Table [I](#S5.T1 "TABLE I ‣ V Results & Discussion ‣ Scaling Up Adaptive Filter Optimizers").
We attempt to scale up our baselines and then do so with our proposed
model. When scaling model size, we notice that scaling the unsupervised
model from S to L (S$`\cdot`$U$`\cdot`$P to L$`\cdot`$U$`\cdot`$P)
results in a peak gain of $`\approx 1`$dB, to $`8.03`$dB ERLE. In
contrast, scaling the supervised model from S to L
(S$`\cdot`$S$`\cdot`$P to L$`\cdot`$S$`\cdot`$P) yields larger gains,
peaking at $`11.62`$dB. When scaling optimization steps, we find the
unsupervised models from S$`\cdot`$U$`\cdot`$P to
S$`\cdot`$U$`\cdot`$PUx2 results in marginal or even a negative
performances changes. In contrast, scaling from S$`\cdot`$S$`\cdot`$P to
S$`\cdot`$S$`\cdot`$PUx2 provides $`+1`$dB, and L$`\cdot`$S$`\cdot`$P to
L$`\cdot`$S$`\cdot`$PUx2 provides $`+2.65`$dB, showing supervision is
crucial to unlock the benefit of multiple opt. steps per frame. Our
best$`\cdot`$performing L$`\cdot`$S$`\cdot`$PUx2 scores over $`14`$dB
ERLE, doubling the S$`\cdot`$U$`\cdot`$P performance of $`7.09`$dB.

When benchmarking against competing methods, we note that SOTA
sueprvised NKF method is most comparable. Our S$`\cdot`$S$`\cdot`$PU
model matches NKF performance while using only one-fifth of the NKF
MFLOP count. Our M$`\cdot`$S$`\cdot`$PU model further enhances all
metrics and uses fewer MFLOPs. For our best-performing
L$`\cdot`$S$`\cdot`$PUx2, we score $`14.25`$dB ERLE, a $`4.96`$dB
improvement over NKF. In perceptual metrics, our top-performing
L$`\cdot`$S$`\cdot`$PUx2 model achieves $`11`$dB in R-ERLE and a
R-AEC-MOS of $`3.94`$, while remaining real-time on a single CPU core.
Surprisingly, RTF scales non-linearly with MFLOPs and model size,
showing untapped scaling potential.

|                          |     |                    |     |                 |     |                 |     |                  |     |                      |     |                   |
| ------------------------ | --- | ------------------ | --- | --------------- | --- | --------------- | --- | ---------------- | --- | -------------------- | --- | ----------------- |
| Model                    |     | SI-SDR$`\uparrow`$ |     | SIR$`\uparrow`$ |     | SAR$`\uparrow`$ |     | STOI$`\uparrow`$ |     | MFLOPs$`\downarrow`$ |     | RTF$`\downarrow`$ |
| Mixture                  |     | -0.71              |     | \-              |     | \-              |     | 0.674            |     | \-                   |     | \-                |
| NLMS$`\cdot`$P           |     | 8.60               |     | 16.21           |     | 9.78            |     | 0.905            |     | 0.43                 |     | 0.36              |
| NLMS$`\cdot`$PU          |     | 8.84               |     | 16.54           |     | 10.00           |     | 0.910            |     | 0.47                 |     | 0.47              |
| RLS$`\cdot`$P            |     | 9.84               |     | 16.70           |     | 9.70            |     | 0.919            |     | 0.53                 |     | 0.50              |
| RLS$`\cdot`$PU           |     | 10.14              |     | 17.16           |     | 11.49           |     | 0.924            |     | 0.54                 |     | 0.62              |
| S$`\cdot`$U$`\cdot`$P    |     | 12.20              |     | 22.57           |     | 12.79           |     | 0.931            |     | 4.70                 |     | 0.41              |
| M$`\cdot`$U$`\cdot`$P    |     | 12.62              |     | 22.56           |     | 13.26           |     | 0.938            |     | 12.08                |     | 0.45              |
| L$`\cdot`$U$`\cdot`$P    |     | 12.45              |     | 22.43           |     | 13.09           |     | 0.935            |     | 36.35                |     | 0.53              |
| S$`\cdot`$S$`\cdot`$P    |     | 13.92              |     | 23.00           |     | 14.66           |     | 0.950            |     | 4.70                 |     | 0.41              |
| M$`\cdot`$S$`\cdot`$P    |     | 14.34              |     | 23.45           |     | 15.07           |     | 0.953            |     | 12.08                |     | 0.45              |
| L$`\cdot`$S$`\cdot`$P    |     | 14.69              |     | 24.36           |     | 15.33           |     | 0.954            |     | 36.35                |     | 0.53              |
| S$`\cdot`$S$`\cdot`$PU   |     | 15.46              |     | 25.69           |     | 16.09           |     | 0.956            |     | 4.74                 |     | 0.51              |
| M$`\cdot`$S$`\cdot`$PU   |     | 16.83              |     | 27.70           |     | 17.41           |     | 0.960            |     | 12.12                |     | 0.54              |
| L$`\cdot`$S$`\cdot`$PU   |     | 17.22              |     | 28.37           |     | 17.80           |     | 0.962            |     | 36.39                |     | 0.62              |
| S$`\cdot`$S$`\cdot`$PUx2 |     | 15.67              |     | 25.89           |     | 16.35           |     | 0.956            |     | 9.07                 |     | 0.70              |
| M$`\cdot`$S$`\cdot`$PUx2 |     | 17.06              |     | 28.42           |     | 17.63           |     | 0.961            |     | 23.83                |     | 0.76              |
| L$`\cdot`$S$`\cdot`$PUx2 |     | 17.72              |     | 29.52           |     | 18.25           |     | 0.964            |     | 72.37                |     | 0.91              |

TABLE II: Beaforming performance vs. computational cost.

### V-C GSC Beamforming Scaling and Benchmarking

Beamforming results are
in Table [II](#S5.T2 "TABLE II ‣ V-B AEC Scaling Ablation and Benchmarking ‣ V Results & Discussion ‣ Scaling Up Adaptive Filter Optimizers").
Notably, SMS-AF improvements apply without any modifications. Here, all
models assume access to a steering vector, which can be challenging to
estimate in practice. Again, supervision and multi-step optimization
yield significant performance gains. Our S$`\cdot`$S$`\cdot`$P model
outperforms all baselines including L$`\cdot`$U$`\cdot`$P across all
metrics. Our model scales reliably with the L$`\cdot`$S$`\cdot`$P
variant improving performance in all metrics. Scaling up the iterations
to S$`\cdot`$U$`\cdot`$PUx2 yields larger gains across all metrics. Our
largest and best model, L$`\cdot`$S$`\cdot`$PUx2 scores a remarkable
17.72 dB SI-SDR while still being real-time. Of note, the
L$`\cdot`$S$`\cdot`$PU model has the same RTF as RLS$`\cdot`$PU, even
though RLS uses fewer operations. Again, we show that SMS-AF performance
scales with both model capacity and optimization steps per frame.

## VI Conclusion

We introduce a method for a neural network-based adaptive filter
optimizers called supervised multi-step adaptive filters (SMS-AF). We
extend meta-adaptive filtering methods with several advances that
combine to reliably increase performance by leveraging more computation.
We evaluate our method on low latency, online AEC and GSC tasks, compare
against many baselines and test on both synthetic and real data. SMS-AF
improves both subjective and objective metrics, achieving
$`\approx 5`$dB ERLE/SI-SDR gains compared to prior work, and increases
the performance ceiling across AEC and GSC. Furthermore, we relate our
work to the Kalman filter and meta-AFs, giving insight for many other
applications. We believe scaling-up AFs is a promising direction and
hope our results encourage future work on scalable, general purpose AFs.

## References

- \[1\] Bernard Widrow and Marcian E. Hoff, “Adaptive switching
  circuits,” Tech. Rep., Stanford University, 1960.
- \[2\] V. John Mathews, “Adaptive polynomial filters,” IEEE Signal
  Processing Magazine (SPM), 1991.
- \[3\] José Antonio Apolinário, José Antonio Apolinário, and
  R Rautmann, QRD-RLS adaptive filtering, Springer, 2009.
- \[4\] Simon S. Haykin, Adaptive filter theory, Pearson, 2008.
- \[5\] Lawrence R. Rabiner, Bernard Gold, and CK Yuen, Theory and
  application of digital signal processing, Prentice-Hall, 2016.
- \[6\] Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger,
  “Broken neural scaling laws,” in International Conference on Learning
  Representations (ICLR), 2022.
- \[7\] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena
  Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas,
  Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al., “Training
  compute-optimal large language models,” arXiv preprint
  arXiv:2203.15556, 2022.
- \[8\] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer,
  Trevor Darrell, and Saining Xie, “A ConvNet for the 2020s,” in
  IEEE/CVF Conference on Computer Vision and Pattern Recognition
  (CVPR), 2022.
- \[9\] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli
  Shechtman, Sylvain Paris, and Taesung Park, “Scaling up gans for
  text-to-image synthesis,” in IEEE/CVF Conference on Computer Vision
  and Pattern Recognition (CVPR), 2023.
- \[10\] Jonah Casebeer, Jacob Donley, Daniel Wong, Buye Xu, and Anurag
  Kumar, “NICE-Beam: Neural integrated covariance estimators for
  time-varying beamformers,” arXiv:2112.04613, 2021.
- \[11\] Thomas Haubner, Andreas Brendel, and Walter Kellermann,
  “End-to-end deep learning-based adaptation control for
  frequency-domain adaptive system identification,” in IEEE
  International Conference on Acoustics, Speech and Signal Processing
  (ICASSP), 2022.
- \[12\] Thomas Haubner and Walter Kellermann, “Deep learning-based
  joint control of acoustic echo cancellation, beamforming and
  postfiltering,” in IEEE European Signal Processing Conference
  (EUSIPCO), 2022.
- \[13\] Amir Ivry, Israel Cohen, and Baruch Berdugo, “Deep adaptation
  control for acoustic echo cancellation,” in IEEE International
  Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
- \[14\] Behrad Soleimani, Henning Schepker, and Majid Mirbagheri,
  “Neural-afc: Learning-based step-size control for adaptive feedback
  cancellation with closed-loop model training,” in IEEE International
  Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023.
- \[15\] Dong Yang, Fei Jiang, Wei Wu, Xuefei Fang, and Muyong Cao,
  “Low-complexity acoustic echo cancellation with neural kalman
  filtering,” in IEEE International Conference on Acoustics, Speech and
  Signal Processing (ICASSP), 2023.
- \[16\] Jonah Casebeer, Nicholas J. Bryan, and Paris Smaragdis,
  “Auto-DSP: Learning to optimize acoustic echo cancellers,” in IEEE
  Workshop on Applications of Signal Processing to Audio and Acoustics
  (WASPAA), 2021.
- \[17\] Jonah Casebeer, Nicholas J. Bryan, and Paris Smaragdis,
  “Meta-AF: Meta-learning for adaptive filters,” IEEE Transactions on
  Audio, Speech, and Language Processing (TASLP), 2022.
- \[18\] Junkai Wu, Jonah Casebeer, Nicholas J. Bryan, and Paris
  Smaragdis, “Meta-learning for adaptive filters with higher-order
  frequency dependencies,” in IEEE International Workshop on Acoustic
  Signal Enhancement (IWAENC), 2022.
- \[19\] Jonah Casebeer, Junkai Wu, and Paris Smaragdis, “Meta-af echo
  cancellation for improved keyword spotting,” arXiv:2312.10605, 2023.
- \[20\] Marcin Andrychowicz, Misha Denil, Sergio Gómez Colmenarejo,
  Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and
  Nando de Freitas, “Learning to learn by gradient descent by gradient
  descent,” in NeurIPS, 2016.
- \[21\] Moritz Wolter and Angela Yao, “Complex gated recurrent neural
  networks,” in NeurIPS, 2018.
- \[22\] Gerald Enzner and Peter Vary, “Frequency-domain adaptive Kalman
  filter for acoustic echo control in hands-free telephones,” Elsevier
  Signal Processing, 2006.
- \[23\] Ross Cutler, Ando Saabas, Tanel Parnamaa, Marju Purin, Hannes
  Gamper, Sebastian Braun, Karsten Sorensen, and Robert Aichner, “ICASSP
  2022 acoustic echo cancellation challenge,” in IEEE International
  Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
- \[24\] Gerald Enzner, Herbert Buchner, Alexis Favrot, and Fabian
  Kuech, “Acoustic echo control,” in Academic press library in signal
  processing. Elsevier, 2014.
- \[25\] Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji
  Watanabe, “The third CHiME speech separation and recognition
  challenge: Dataset, task and baselines,” in Automatic Speech
  Recongition and Understanding Workshop (ASRU). IEEE, 2015.
- \[26\] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, and John R.
  Hershey, “SDR–half-baked or well done?,” in IEEE International
  Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019.
- \[27\] Emmanuel Vincent, Rémi Gribonval, and Cédric Févotte,
  “Performance measurement in blind audio source separation,” IEEE
  Transactions on Audio, Speech, and Language Processing (TASLP), 2006.
- \[28\] Cees H. Taal, Richard C. Hendriks, Richard Heusdens, and Jesper
  Jensen, “An algorithm for intelligibility prediction of time–frequency
  weighted noisy speech,” IEEE Transactions on Audio, Speech, and
  Language Processing (TASLP), 2011.
