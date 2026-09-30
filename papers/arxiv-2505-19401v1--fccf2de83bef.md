---
identifier: arxiv:2505.19401v1
title: "Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement"
authors:
  - Jangyeon Kim
  - Ui-Hyeop Shin
  - Jaehyun Ko
  - Hyung-Min Park
published: "2025-05-26T01:34:53+00:00"
url: https://arxiv.org/abs/2505.19401v1
source: arxiv
doi: null
arxiv_id: 2505.19401v1
categories:
  - eess.AS
---

Kim Shin Ko Park Sogang UniversityRepublic of Korea Sogang
UniversityRepublic of Korea

# Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement

Jangyeon    Ui-Hyeop    Jaehyun    Hyung-Min Affiliation: Department of
Artificial Intelligence Affiliation: Department of Electronic
Engineering

###### Abstract

This paper presents an efficient speech enhancement (SE) approach that
reuses a processing block repeatedly instead of conventional stacking.
Rather than increasing the number of blocks for learning deep latent
representations, repeating a single block leads to progressive
refinement while reducing parameter redundancy. We also minimize domain
transformation by keeping an encoder and decoder shallow and reusing a
single sequence modeling block. Experimental results show that the
number of processing stages is more critical to performance than the
number of blocks with different weights. Also, we observed that the
proposed method gradually refines a noisy input within a single block.
Furthermore, with the block reuse method, we demonstrate that deepening
the encoder and decoder can be redundant for learning deep complex
representation. Therefore, the experimental results confirm that the
proposed block reusing enables progressive learning and provides an
efficient alternative for SE.

###### keywords

speech enhancement, block reusing, progressive refinement, dual-path
architecture, parameter efficiency

^(†)^(†)email: {jykim97, dmlguq123, jhko, hpark}@sogang.ac.kr

## 1 Introduction

^(†)^(†)footnotetext: \*Equal contribution.^(†)^(†)footnotetext:
^(†)Corresponding Author.

With the rapid development of deep learning, speech enhancement (SE) has
been significantly improved as a critical pre-processing step in various
applications. In particular, from the conventional real-valued masking
in time-frequency (TF) with Short-Time Fourier Transform (STFT), the
estimation of complex mask has led to substantial improvement of
SE \[[1](#bib.bib1), [2](#bib.bib2)\]. Furthermore, from U-Net-based
architectures \[[1](#bib.bib1), [2](#bib.bib2), [3](#bib.bib3),
[4](#bib.bib4), [5](#bib.bib5)\], dual-path modeling in the TF domain
further boosted performance in both speech separation
(SS) \[[6](#bib.bib6), [7](#bib.bib7), [8](#bib.bib8)\] and
SE \[[9](#bib.bib9), [10](#bib.bib10), [11](#bib.bib11),
[12](#bib.bib12)\]. More recently, in addition to mask-based approaches,
studies based on direct spectral mapping also has been explored, which
eliminates the need for masking by outputting enhanced signals either
partially \[[11](#bib.bib11), [12](#bib.bib12)\] or in
full \[[7](#bib.bib7), [8](#bib.bib8)\].

Progress in dual-path modeling has been driven by the development of
sequence modeling capacity of Transformer-based
architectures \[[8](#bib.bib8), [9](#bib.bib9), [10](#bib.bib10),
[11](#bib.bib11)\]. Transformer-based dual-path models have demonstrated
remarkable performance improvements by stacking multiple layers. At the
same time, while model performance has continued to improve, there has
been a notable shift where model sizes have generally decreased, whereas
computational complexity has increased. This trend is evident in the
transition from U-Net-based architectures to Transformer-based TF
models. It is also reported that the extent of feature processing,
influenced by computational complexity, plays a more critical role than
the complexity of domain transitions between different representations
with a number of parameters and non-linear projections in the
model \[[13](#bib.bib13)\].

Based on this observation, we consider an alternative approach where we
reuse a processing block repeatedly instead of stacking multiple
processing blocks. These weight sharing across layers have been actively
explored in automatic speech recognition \[[14](#bib.bib14),
[15](#bib.bib15)\] or natural language processing \[[16](#bib.bib16),
[17](#bib.bib17)\]. However, since these tasks often involve
transforming input into different domains, sharing weights can lead to
performance limitations \[[18](#bib.bib18)\]. To mitigate this, partial
weight sharing \[[14](#bib.bib14)\] or adapter-based
mechanisms \[[18](#bib.bib18), [19](#bib.bib19), [20](#bib.bib20)\] have
been introduced. On the other hand, in SS, such iterative block reusing
technique has been successfully applied \[[21](#bib.bib21),
[22](#bib.bib22)\] and shown to be effective. This is likely because SS
operates within the same domain for both input and output. When the
input and output domains remain the same, we hypothesize that
progressive refinement through iterative block reuse can serve as a more
efficient alternative to learning deep latent representations using a
model with multiple blocks.

Motivated by this, we introduce the block reusing method in SE. Compared
to conventional stacking, reusing a block does not result in significant
performance degradation in SE. Furthermore, we investigated the impact
of the number of stacked and repeated blocks. Empirically, the number of
processing stages itself plays a more critical role for performance than
the number of blocks with unique weights. Also, our experiments show
that training with a repeated block naturally leads a single block to
progressively refine the input in contrast to conventional stacked
architectures. Moreover, a deep encoder-decoder structure for learning
deeper latent representations can be redundant. To validate our
hypothesis, we selected MP-SENet \[[11](#bib.bib11)\] as a baseline
which has a convolutional encoder-decoder (CED) architecture that has
been widely adopted in recent models \[[1](#bib.bib1), [2](#bib.bib2),
[5](#bib.bib5), [10](#bib.bib10)\]. Then, we minimized domain
transformation by keeping CED shallow while increasing the repetition of
a single Transformer-based dual-path block in the intermediate layer.
Notably, the proposed model retains competitive performance despite the
reduction in parameters and computational complexity. This demonstrates
that reusing blocks enables progressive feature refinement while
significantly reducing parameters, making it an efficient approach for
SE tasks without relying on deep latent features learned through a deep
encoder and decoder.

![Refer to caption](2505.19401v1/stacking_v2.png)

(a)

![Refer to caption](2505.19401v1/repeating_v2.png)

(b)

Figure 1: Comparison of block (a) stacking and (b) repeating

## 2 The proposed method

The SE task fundamentally involves generating an output signal that
retains only the desired speech components while remaining in the same
domain as the input. Therefore, rather than learning complex
representations with multiple blocks, a progressive refinement approach
through iterative processing with a single block can serve as an
effective strategy.

### 2.1 Overview of SE network

As illustrated in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement")(a),
conventional SE methods are often based on the CED architecture with
stacked sequence modeling blocks \[[10](#bib.bib10), [11](#bib.bib11),
[12](#bib.bib12)\]. When a noisy signal is given in the TF domain with
$`T`$ frames and $`F`$ frequency bins, the noisy input is encoded as a
feature representation with the shape of
$`{C\hskip-1.42262pt\times\hskip-1.42262ptT\hskip-1.42262pt\times\hskip-2.84526ptF}`$
in TF domain, where $`C`$ is the feature dimension. The encoded feature
is then processed by a stack of $`B`$ sequence modeling blocks. Finally,
the decoder transforms the output feature back to its original shape to
estimate either the mask value or the direct speech signal.

### 2.2 The proposed block reusing method

Instead of stacking multiple blocks, we propose using a single sequence
modeling block repeatedly as shown in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement")(b),
which may lead to progressive enhancement without explicit intermediate
loss. In particular, this strategy, often referred to as the unfolding
method, can have some variants depending on the information fusion
scheme between processing stages \[[21](#bib.bib21)\]. When using the
output of a shared block as input in subsequent iterations, we can
simply reuse the output of the previous stage as input to the next
stages as a direct connection (DC). Meanwhile, we can also add initial
feature to the each stage output as a summation connection (SC) to
encourage leveraging the input feature. On the other hand, simply
reusing the single block can be challenging for more complex noisy
inputs. Therefore, we can consider repeating stacks of blocks to enhance
modeling capacity of sequence block.

### 2.3 Dual-path TF model for SE

To validate and analyze our block reuse method, we selected
TF-Locoformer \[[8](#bib.bib8)\] and MP-SENet \[[11](#bib.bib11)\] as
baseline models for our experiments. Both networks use dual-path TF
blocks to model the input features in the TF domain. The dual-path TF
blocks alternately process the feature along the time and frequency
dimensions to capture temporal and inter-frequency dependencies,
respectively, as shown in
Figure [2](#S2.F2 "Figure 2 ‣ 2.3 Dual-path TF model for SE ‣ 2 The proposed method ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement").

![Refer to caption](2505.19401v1/loco.png)

(a)

![Refer to caption](2505.19401v1/mpse.png)

(b)

Figure 2: Two baseline networks using dual-path TF model

#### 2.3.1 TF-Locoformer

The recently proposed TF-Locoformer \[[8](#bib.bib8)\] achieved
impressive performance in SS and SE. As in
Figure [2](#S2.F2 "Figure 2 ‣ 2.3 Dual-path TF model for SE ‣ 2 The proposed method ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement")(a),
it follows a CED framework, consisting of only a single Conv2D and
Deconv2D layer in the encoder and decoder, respectively. From the real
and imaginary (RI) components of a noisy input, TF-Locoformer directly
estimates the RI components of the enhanced signal. Also, as a unit
module for the dual-path TF block, TF-Locoformer uses a modified
feed-forward network to enhance local modeling based on convolutional
layers with a Swish-gated linear unit (Conv-SwiGLU) as macaron-style
structure in Transformer. Refer to \[[8](#bib.bib8)\] for detailed
structure.

#### 2.3.2 MP-SENet

As depicted in
Figure [2](#S2.F2 "Figure 2 ‣ 2.3 Dual-path TF model for SE ‣ 2 The proposed method ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement")(b),
MP-SENet combines magnitude masking and phase mapping based on the CED
architecture, where the encoder consists of a cascade of a convolutional
block, a dilated DenseNet \[[23](#bib.bib23)\], and another
convolutional block, while the two parallel decoders are composed of a
dilated DenseNet followed by a deconvolutional block for magnitude
masking and phase mapping, respectively. As a unit for the dual-path TF
block, MP-SENet uses Conformer \[[24](#bib.bib24)\] where a
convolutional module is incorporated into a macaron-style Transformer to
enhance local modeling capacity.

## 3 Experimental setups

### 3.1 Datasets and evaluation

To train and evaluate the proposed method, we used the Interspeech
DNS-Challenge 2020 dataset \[[25](#bib.bib25)\] for TF-Locoformer and
the VoiceBank+DEMAND dataset \[[26](#bib.bib26)\] for MP-SENet. The DNS
dataset consists of 500 hours of clean speech data and over 180 hours of
noise data. We generated training data by mixing clean speech with noise
at signal-to-noise ratios (SNRs) ranging from -5 to 15 dB. For
evaluation, we used the DNS non-blind test set without reverberation
with SNRs ranging from 0 to 25 dB. Perceptual evaluation of speech
quality (PESQ), short-time objective intelligibility (STOI), and
scale-invariant signal-to-distortion ratio (SI-SDR) were used as
evaluation metrics. The VoiceBank+DEMAND dataset includes 11,572
training utterances from 28 speakers with SNRs of 0, 5, 10, and 15 dB,
and 824 test utterances from 2 speakers with SNRs of 2.5, 7.5, 12.5, and
17.5 dB. For the VoiceBank+DEMAND dataset, the evaluation metrics
include PESQ, segmental SNR (SSNR), and STOI. As additional metrics,
CSIG, CBAK, and COVL were employed to assess different aspects of
perceptual quality \[[10](#bib.bib10)\]. In all experiments, the
sampling rate was set to 16 kHz. We also compared the parameter size and
the number of multiply-accumulate operations (MACs) for 1-second input.

|       |       |        |       |      |      |       |        |
| ----- | ----- | ------ | ----- | ---- | ---- | ----- | ------ |
| $`B`$ | $`R`$ | Param. | MACs  | PESQ |      | STOI  | SI-SDR |
|       |       | (M)    | (G/s) | -WB  | -NB  | (%)   | (dB)   |
| Noisy |       | -      | -     | 1.58 | 2.45 | 91.52 | 9.07   |
| 1     | 1     | 0.5    | 8.7   | 2.82 | 3.28 | 96.51 | 18.46  |
| 4     | 1     | 1.9    | 34.8  | 3.41 | 3.74 | 98.10 | 20.70  |
| 8     | 1     | 3.7    | 69.6  | 3.47 | 3.79 | 98.26 | 21.07  |
| 12    | 1     | 5.6    | 104.4 | 3.49 | 3.81 | 98.31 | 21.32  |
| 16    | 1     | 7.4    | 139.2 | 3.55 | 3.86 | 98.41 | 21.67  |
| 1     | 4     | 0.5    | 34.8  | 3.26 | 3.64 | 97.78 | 19.88  |
| 1     | 8     | 0.5    | 69.6  | 3.39 | 3.73 | 98.08 | 20.51  |
| 1     | 12    | 0.5    | 104.4 | 3.43 | 3.76 | 98.18 | 20.91  |
| 1     | 16    | 0.5    | 139.2 | 3.43 | 3.79 | 98.21 | 20.85  |
| 2     | 8     | 0.9    | 139.2 | 3.48 | 3.81 | 98.31 | 21.06  |
| 4     | 4     | 1.9    | 139.2 | 3.52 | 3.83 | 98.38 | 21.32  |
| 8     | 2     | 3.7    | 139.2 | 3.46 | 3.80 | 98.31 | 21.12  |

Table 1: Evaluation on DNS dataset with various configuration of $`B`$
and $`R`$ in TF-Locoformer. The best performance is highlighted in bold,
and the second-best performance is underlined.

Figure 3: Plot of PESQ-WB vs. MACs on combinations of $`B`$ and $`R`$
values from
Table [1](#S3.T1 "Table 1 ‣ 3.1 Datasets and evaluation ‣ 3 Experimental setups ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement").
The size of each circle is proportional to the parameter size.

### 3.2 Training and model configuration

All models were trained using the AdamW optimizer \[[27](#bib.bib27)\]
for 100 epochs. TF-Locoformer was trained using 4-second segments with a
batch size of 1. STFT was computed using a Hanning window of size 256
and a hop size of 128. The $`C`$ was set to 64. A hidden dimension and a
kernel size are set to 172 and 3 in the Conv-SwiGLU
module \[[8](#bib.bib8)\], respectively. The head of multi-head
self-attention is 4. For training, time-domain $`L_{1}`$ loss and
TF-domain multi-resolution $`L_{1}`$ loss \[[28](#bib.bib28)\] was
utilized. Note that input normalization and the scaling factor in the
loss function from the original work \[[8](#bib.bib8)\] were applied
only in Section 4.4. On the other hand, MP-SENet was trained with
2-second segments with a batch size of 2. We followed the same model
configuration and training procedure as in original work, including
STFT \[[11](#bib.bib11)\]. However, note that the metric discriminator
was not used for model training.

## 4 Results

### 4.1 Impact of varying the number of blocks and repeats

Based on TF-Locoformer, we first explored how varying the number of
blocks $`B`$ and repetitions $`R`$ influences the model’s performance on
DNS dataset. Note that all results were derived using the DC method for
the information fusion scheme. In
Table [1](#S3.T1 "Table 1 ‣ 3.1 Datasets and evaluation ‣ 3 Experimental setups ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement")
and
Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Datasets and evaluation ‣ 3 Experimental setups ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement"),
we observe that as $`B`$ increases, the results consistently improve
along with the corresponding increase in parameters. Notably, with a
single block $`B\hskip-1.42262pt=\hskip-1.42262pt1`$ that results in
only 0.5M parameters, increasing $`R`$ leads to significant performance
improvements across all metrics. To further explore the trade-off
between parameters and model performance through constrained computation
costs, we conducted experiments with various combinations of $`B`$ and
$`R`$ with $`B\hskip-2.27621pt\times\hskip-2.27621ptR`$ fixed to 16.
Although performance varies depending on the combinations of $`B`$ and
$`R`$, their differences are not significant because the total number of
processing stages $`B\hskip-1.42262pt\times\hskip-1.42262ptR`$ is
constant. In particular, it is noteworthy that the combination of
$`B\hskip-1.42262pt=\hskip-1.42262pt4`$ and
$`R\hskip-1.42262pt=\hskip-1.42262pt4`$ (denoted as $`B4R4`$) achieved
performance comparable to that of the $`B16R1`$ configuration, while
using about one-fourth of the parameters. This highlights the
effectiveness of the proposed method, demonstrating that it can achieve
competitive results with significantly fewer parameters than
conventional stacked architectures.

Figure 4: Plots of PESQ-WB and SI-SDR results of intermediate outputs
for B16R1, B8R1, B1R16 and B1R8 on DNS dataset.

![Refer to caption](2505.19401v1/Progressive_B.png)

(a)

![Refer to caption](2505.19401v1/Progressive_R.png)

(b)

Figure 5: Spectrograms of a sample utterance from the DNS dataset at
different processing stages.

### 4.2 Visualization of progressive enhancement by block reusing

To analyze the behavior of the network that reuses a single block
compared to a network with multiple blocks, we plotted the evaluation
results of the intermediate outputs indexed by
$`1\hskip-1.42262pt\leq\hskip-1.42262ptb\hskip-1.42262pt\leq\hskip-1.42262ptB`$
in $`B16R1`$ and $`B8R1`$, and
$`1\hskip-1.42262pt\leq\hskip-1.42262ptr\hskip-1.42262pt\leq\hskip-1.42262ptR`$
in $`B1R16`$ and $`B1R8`$ in
Figure [4](#S4.F4 "Figure 4 ‣ 4.1 Impact of varying the number of blocks and repeats ‣ 4 Results ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement")
while using the same decoder in each network. $`B1R16`$ and $`B1R8`$
inherently learn to enhance the signal progressively by repeatedly using
a single block, starting from a noisy input level. This progressive
refinement can also be observed well in the sample spectrogram in
Figure [5](#S4.F5 "Figure 5 ‣ 4.1 Impact of varying the number of blocks and repeats ‣ 4 Results ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement"),
which is consistent with the results in
Figure [4](#S4.F4 "Figure 4 ‣ 4.1 Impact of varying the number of blocks and repeats ‣ 4 Results ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement").
In contrast, the intermediate outputs in $`B16R1`$ and $`B8R1`$ do not
show enhancement of the noisy input until
$`b\hskip-1.42262pt=\hskip-1.42262pt10`$ and
$`b\hskip-1.42262pt=\hskip-1.42262pt5`$, respectively. This may be due
to the functional partitioning of $`B`$ blocks into encoder and decoder
roles, which could enable the network to learn the deep latent features.

### 4.3 Comparison of fusion schemes in block reusing

|           |        |      |      |       |        |
| --------- | ------ | ---- | ---- | ----- | ------ |
| System    | Fusion | PESQ |      | STOI  | SI-SDR |
|           | scheme | -WB  | -NB  | (%)   | (dB)   |
| $`B16R1`$ | -      | 3.55 | 3.86 | 98.41 | 21.67  |
| $`B1R16`$ | DC     | 3.43 | 3.79 | 98.21 | 20.85  |
|           | SC     | 3.46 | 3.80 | 98.21 | 21.13  |
| $`B4R4`$  | DC     | 3.52 | 3.83 | 98.38 | 21.32  |
|           | SC     | 3.56 | 3.87 | 98.46 | 21.84  |

Table 2: Comparison of fusion schemes in block reusing.

In the network with block reuse, we compared two fusion schemes of DC
and SC.
Table [2](#S4.T2 "Table 2 ‣ 4.3 Comparison of fusion schemes in block reusing ‣ 4 Results ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement")
evaluates the case of $`B1R16`$ and $`B4R4`$. The results indicate that
the SC method achieves better performance than the DC method by
repeatedly adding the input feature, which is consistent with the
findings in \[[21](#bib.bib21)\]. The model $`B4R4`$ with SC method
outperformed the model $`B16R1`$ as the conventional method, with much
fewer parameters (7.4M vs. 1.9M). These results highlight the efficiency
of the proposed approach in SE.

### 4.4 Comparison with existing methods

|                                  |        |       |      |      |       |        |
| -------------------------------- | ------ | ----- | ---- | ---- | ----- | ------ |
| System                           | Param. | MACs  | PESQ |      | STOI  | SI-SDR |
|                                  | (M)    | (G/s) | -WB  | -NB  | (%)   | (dB)   |
| Noisy                            |        |       | 1.58 | 2.45 | 91.52 | 9.07   |
| FullSubNet \[[29](#bib.bib29)\]  | 5.6    | 31.4  | 2.78 | 3.31 | 96.11 | 17.29  |
| CTSNet \[[30](#bib.bib30)\]      | 4.4    | 5.6   | 2.94 | 3.42 | 96.21 | 16.69  |
| TaylorSENet \[[31](#bib.bib31)\] | 5.4    | 6.2   | 3.22 | 3.59 | 97.36 | 19.15  |
| FRCRN \[[4](#bib.bib4)\]         | 6.9    | 242.0 | 3.23 | 3.60 | 97.69 | 19.78  |
| MFNet \[[5](#bib.bib5)\]         | 6.1    | 6.1   | 3.43 | 3.74 | 97.98 | 20.31  |
| USES \[[32](#bib.bib32)\]        | 3.1    | 65.3  | 3.46 | -    | 98.1  | 21.2   |
| TF-Locoformer \[[8](#bib.bib8)\] | 15.0   | 255.1 | 3.72 | -    | 98.8  | 23.3   |
| $`B1R16`$-SC                     | 0.5    | 139.2 | 3.53 | 3.85 | 98.37 | 21.71  |
| $`B4R4`$-SC                      | 1.9    | 139.2 | 3.64 | 3.92 | 98.54 | 22.12  |

Table 3: Comparison of proposed model with previous models on DNS
datasets.

We compared the proposed networks with previous studies on the DNS
dataset. For a fair comparison, the proposed networks were trained by
applying input normalization and a scaling factor $`\ddot{\alpha}`$ in
the loss function \[[8](#bib.bib8)\]. In
Table [3](#S4.T3 "Table 3 ‣ 4.4 Comparison with existing methods ‣ 4 Results ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement"),
$`B1R16`$-SC with only 0.5M parameters achieves superior or competitive
performance compared to most existing methods, despite its minimum
parameters. Next, we examine the $`B4R4`$-SC model, which demonstrates
the highest parameter efficiency and performance among the proposed
methods. By a larger margin, this model surpasses most prior works.
Although the proposed models show slightly lower performance compared
with TF-Locoformer \[[8](#bib.bib8)\], it is noteworthy that the
proposed models maintain much smaller model sizes and about half the
computational costs, while still delivering high performance.

### 4.5 Investigation of deep encoder and decoder

|                                                    |       |        |       |      |       |       |      |      |      |
| -------------------------------------------------- | ----- | ------ | ----- | ---- | ----- | ----- | ---- | ---- | ---- |
| $`B`$                                              | $`R`$ | Param. | MACs  | PESQ | STOI  | SSNR  | CSIG | CBAK | COVL |
|                                                    |       | (M)    | (G/s) | -WB  | (%)   | (dB)  |      |      |      |
| Noisy                                              |       | -      | -     | 1.97 | 91.00 | 1.68  | 3.35 | 2.44 | 2.63 |
| Encoder-decoder with DenseNet of $`K=4`$ (default) |       |        |       |      |       |       |      |      |      |
| 4                                                  | 1     | 2.1    | 43.9  | 3.37 | 95.87 | 10.68 | 4.68 | 3.89 | 4.13 |
| Encoder-decoder with DenseNet of $`K=2`$           |       |        |       |      |       |       |      |      |      |
| 4                                                  | 1     | 1.3    | 27.2  | 3.39 | 95.80 | 10.53 | 4.69 | 3.89 | 4.15 |
| 1                                                  | 4     | 0.6    | 27.2  | 3.27 | 94.51 | 10.46 | 4.60 | 3.83 | 4.02 |
| Encoder-decoder with DenseNet of $`K=1`$           |       |        |       |      |       |       |      |      |      |
| 4                                                  | 1     | 1.1    | 22.4  | 3.29 | 95.43 | 10.00 | 4.64 | 3.81 | 4.06 |
| 1                                                  | 4     | 0.4    | 22.4  | 3.26 | 95.44 | 10.25 | 4.62 | 3.81 | 4.03 |
| 6                                                  | 1     | 1.5    | 32.0  | 3.41 | 96.05 | 10.59 | 4.74 | 3.90 | 4.18 |
| 1                                                  | 6     | 0.4    | 32.0  | 3.36 | 95.65 | 10.21 | 4.68 | 3.86 | 4.13 |

Table 4: Evaluation on VoiceBank+DEMAND dataset with various
configuration of $`K`$, $`B`$, and $`R`$ in MP-SENet.

Finally, we examine the impact of varying the depth $`K`$ of the dilated
DenseNet \[[23](#bib.bib23)\] in the encoder and decoder of MP-SENet,
which is directly related to the degree of domain transformation. In
Table [4](#S4.T4 "Table 4 ‣ 4.5 Investigation of deep encoder and decoder ‣ 4 Results ‣ Stack Less, Repeat More: A Block Reusing Approach for Progressive Speech Enhancement"),
starting from the default setting of
$`K\hskip-1.42262pt=\hskip-1.42262pt4`$, we evaluated the cases of
$`K\hskip-1.42262pt=\hskip-1.42262pt2`$ and
$`K\hskip-1.42262pt=\hskip-1.42262pt1`$. When
$`B\hskip-1.42262pt=\hskip-1.42262pt4`$, reducing $`K`$ significantly
decreases both the parameters and computational cost. In particular,
compared to $`K\hskip-1.42262pt=\hskip-1.42262pt4`$, the case of
$`K\hskip-1.42262pt=\hskip-1.42262pt1`$ requires nearly half of both the
parameters and computational cost. In spite of the reduced model size
and complexity, the performance degradation observed for
$`K\hskip-1.42262pt=\hskip-1.42262pt1`$ remains relatively minor. More
notably, the model with $`K\hskip-1.42262pt=\hskip-1.42262pt2`$ achieves
comparable performance to the case of
$`K\hskip-1.42262pt=\hskip-1.42262pt4`$.

With a shallow encoder and decoder with
$`K\hskip-1.42262pt=\hskip-1.42262pt1`$, the case with
$`B\hskip-1.42262pt=\hskip-1.42262pt6`$,
$`R\hskip-1.42262pt=\hskip-1.42262pt1`$ outperforms the default
configuration, achieving the best performance even with fewer parameters
and computations. Additionally, the
$`B\hskip-1.42262pt=\hskip-1.42262pt1`$ and
$`R\hskip-1.42262pt=\hskip-1.42262pt6`$ configuration leads to a
substantial reduction in parameters, with a only slight performance
decrease. In particular, the model with
$`B\hskip-1.42262pt=\hskip-1.42262pt1,R\hskip-1.42262pt=\hskip-1.42262pt6`$
achieves competitive results compared to the default setting, even with
significantly fewer parameters (0.4M). These results consistently
support the idea that complex domain transformation with a deep
encoder-decoder is not essential, whereas the number of processing
iterations plays a more critical role in SE tasks.

## 5 Conclusion

We presented a progressive SE framework that reuses a processing block
efficiently. We confirmed that the number of processing iterations is
more critical than the parameter size. Repeating a single block enables
progressive refinement, leading to effective performance with fewer
parameters. Additionally, we demonstrated that minimizing domain
transformation through a shallow CED and block reuse enables progressive
feature refinement in SE tasks while significantly reducing parameters
without relying on deep latent representations.

However, this work has several limitations that should be addressed in
future research. First, we focused on denoising tasks in non-reverberant
conditions. Future studies should investigate the model’s performance in
reverberant environments to assess its robustness in real-world
scenarios. Also, we experimented using only dual-path TF model. Further
experiments are required on various models to validate the general
performance of the proposed approach. Finally, our study primarily
examined non-causal models. Since SE tasks often require real-time
applications such as hearing-aids and telecommunication systems, future
work should consider causal architectures.

## References

- \[1\] H.-S. Choi, J.-H. Kim, J. Huh, A. Kim, J.-W. Ha, and K. Lee,
  “Phase-aware speech enhancement with deep complex u-net,” in _Proc.
  Int. Conf. Learn. Represent. (ICLR)_, 2018.
- \[2\] Y. Hu, Y. Liu, S. Lv, M. Xing, S. Zhang, Y. Fu, J. Wu, B. Zhang,
  and L. Xie, “DCCRN: Deep Complex Convolution Recurrent Network for
  Phase-Aware Speech Enhancement,” in _Proc. Interspeech_, 2020, pp.
  2472–2476.
- \[3\] Z. Kong, W. Ping, A. Dantrey, and B. Catanzaro, “Speech
  Denoising in the Waveform Domain With Self-Attention,” in _Proc. IEEE
  Int. Conf. Acoust., Speech Signal Process. (ICASSP)_, 2022, pp.
  7867–7871.
- \[4\] S. Zhao, T. H. Nguyen, and B. Ma, “Monaural Speech Enhancement
  with Complex Convolutional Block Attention Module and Joint Time
  Frequency Losses,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal
  Process. (ICASSP)_, 2021, pp. 6648–6652.
- \[5\] L. Liu, H. Guan, J. Ma, W. Dai, G. Wang, and S. Ding, “A Mask
  Free Neural Network for Monaural Speech Enhancement,” in _Proc.
  Interspeech_, 2023, pp. 2468–2472.
- \[6\] L. Yang, W. Liu, and W. Wang, “TFPSNet: Time-Frequency Domain
  Path Scanning Network for Speech Separation,” in _Proc. IEEE Int.
  Conf. Acoust., Speech Signal Process. (ICASSP)_, 2022, pp. 6842–6846.
- \[7\] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and
  S. Watanabe, “TF-GridNet: Integrating Full- and Sub-Band Modeling for
  Speech Separation,” _IEEE/ACM Trans. Audio, Speech, Language
  Process._, vol. 31, pp. 3221–3236, 2023.
- \[8\] K. Saijo, G. Wichern, F. G. Germain, Z. Pan, and J. Le Roux,
  “TF-Locoformer: Transformer with local modeling by convolution for
  speech separation and enhancement,” in _Proc. Int. Workshop Acoust.
  Echo Noise Control (IWAENC)_, 2024, pp. 205–209.
- \[9\] F. Dang, H. Chen, and P. Zhang, “DPT-FSNet: Dual-Path
  Transformer Based Full-Band and Sub-Band Fusion Network for Speech
  Enhancement,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal
  Process. (ICASSP)_, 2022, pp. 6857–6861.
- \[10\] R. Cao, S. Abdulatif, and B. Yang, “CMGAN: Conformer-based
  Metric GAN for Speech Enhancement,” in _Proc. Interspeech_, 2022, pp.
  936–940.
- \[11\] Y.-X. Lu, Y. Ai, and Z.-H. Ling, “MP-SENet: A Speech
  Enhancement Model with Parallel Denoising of Magnitude and Phase
  Spectra,” in _Proc. Interspeech_, 2023, pp. 3834–3838.
- \[12\] R. Chao, W.-H. Cheng, M. La Quatra, S. M. Siniscalchi, C.-H. H.
  Yang, S.-W. Fu, and Y. Tsao, “An investigation of incorporating mamba
  for speech enhancement,” _arXiv preprint arXiv:2405.06573_, 2024.
- \[13\] W. Zhang, K. Saijo, J. weon Jung, C. Li, S. Watanabe, and
  Y. Qian, “Beyond Performance Plateaus: A Comprehensive Study on
  Scalability in Speech Enhancement,” in _Proc. Interspeech_, 2024, pp.
  1740–1744.
- \[14\] S. M. Hernandez, D. Zhao, S. Ding, A. Bruguier,
  R. Prabhavalkar, T. N. Sainath, Y. He, and I. McGraw, “Sharing low
  rank conformer weights for tiny always-on ambient speech recognition
  models,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal Process.
  (ICASSP)_. IEEE, 2023, pp. 1–5.
- \[15\] G. Wei, Z. Duan, S. Li, G. Yang, X. Yu, and J. Li, “Sim-T:
  Simplify the Transformer Network by Multiplexing Technique for Speech
  Recognition,” _arXiv preprint arXiv:2304.04991_, 2023.
- \[16\] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser,
  “Universal transformers,” in _Proc. Int. Conf. Learn. Represent.
  (ICLR)_, 2019.
- \[17\] T. Ge, S.-Q. Chen, and F. Wei, “EdgeFormer: A
  Parameter-Efficient Transformer for On-Device Seq2Seq Generation,” in
  _Proc. Conf. Empir. Methods Nat. Lang. Process. (EMNLP)_, 2022, pp.
  10 786–10 798.
- \[18\] K. Shim, J. Lee, and H. Kim, “Leveraging Adapter for
  Parameter-Efficient ASR Encoder,” in _Proc. Interspeech_, 2024, pp.
  2380–2384.
- \[19\] H. Tang, Z. Liu, C. Zeng, and X. Li, “Beyond universal
  transformer: block reusing with adaptor in transformer for automatic
  speech recognition,” in _Proc. Int. Symp. Neural Networks (ISNN)_,
  2024, pp. 69–79.
- \[20\] Y. Wang and J. Li, “Residualtransformer: Residual Low-Rank
  Learning With Weight-Sharing For Transformer Layers,” in _Proc. IEEE
  Int. Conf. Acoust., Speech Signal Process. (ICASSP)_, 2024, pp.
  11 161–11 165.
- \[21\] X. Hu, K. Li, W. Zhang, Y. Luo, J.-M. Lemercier, and
  T. Gerkmann, “Speech separation using an asynchronous fully recurrent
  convolutional neural network,” _Adv. Neural Inf. Process. Syst._,
  vol. 34, pp. 22 509–22 522, 2021.
- \[22\] K. Li, R. Yang, and X. Hu, “An efficient encoder-decoder
  architecture with top-down attention for speech separation,” in _Proc.
  Int. Conf. Learn. Represent. (ICLR)_, 2023.
- \[23\] A. Pandey and D. Wang, “Densely Connected Neural Network with
  Dilated Convolutions for Real-Time Speech Enhancement in The Time
  Domain,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal Process.
  (ICASSP)_, 2020, pp. 6629–6633.
- \[24\] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu,
  W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer:
  Convolution-augmented transformer for speech recognition,” in _Proc.
  Interspeech_, 2020, pp. 5036–5040.
- \[25\] C. K. Reddy, V. Gopal, R. Cutler, E. Beyrami, R. Cheng,
  H. Dubey, S. Matusevych, R. Aichner, A. Aazami, S. Braun, P. Rana,
  S. Srinivasan, and J. Gehrke, “The INTERSPEECH 2020 Deep Noise
  Suppression Challenge: Datasets, Subjective Testing Framework, and
  Challenge Results,” in _Proc. Interspeech_, 2020, pp. 2492–2496.
- \[26\] C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi,
  “Investigating rnn-based speech enhancement methods for noise-robust
  text-to-speech,” in _Proc. SSW_, 2016, pp. 146–152.
- \[27\] I. Loshchilov and F. Hutter, “Decoupled Weight Decay
  Regularization,” in _Proc. Int. Conf. Learn. Represent. (ICLR)_, 2019.
- \[28\] Y.-J. Lu, S. Cornell, X. Chang, W. Zhang, C. Li, Z. Ni, Z.-Q.
  Wang, and S. Watanabe, “Towards Low-Distortion Multi-Channel Speech
  Enhancement: The ESPNET-Se Submission to the L3DAS22 Challenge,” in
  _Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)_,
  2022, pp. 9201–9205.
- \[29\] X. Hao, X. Su, R. Horaud, and X. Li, “Fullsubnet: A full-band
  and sub-band fusion model for real-time single-channel speech
  enhancement,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal
  Process. (ICASSP)_, 2021, pp. 6633–6637.
- \[30\] A. Li, W. Liu, C. Zheng, C. Fan, and X. Li, “Two heads are
  better than one: A two-stage complex spectral mapping approach for
  monaural speech enhancement,” _IEEE/ACM Trans. Audio, Speech, Language
  Process._, vol. 29, pp. 1829–1843, 2021.
- \[31\] A. Li, S. You, G. Yu, C. Zheng, and X. Li, “Taylor, Can You
  Hear Me Now? A Taylor-Unfolding Framework for Monaural Speech
  Enhancement,” in _Proc. IJCAI_, 2022, pp. 4193–4200.
- \[32\] W. Zhang, K. Saijo, Z.-Q. Wang, S. Watanabe, and Y. Qian,
  “Toward Universal Speech Enhancement For Diverse Input Conditions,” in
  _Proc. IEEE ASRU_, 2023, pp. 1–6.
