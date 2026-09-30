---
identifier: arxiv:2501.06530v1
title: Multi-modal Speech Enhancement with Limited Electromyography Channels
authors:
  - Fuyuan Feng
  - Longting Xu
  - Rohan Kumar Das
published: "2025-01-11T12:33:33+00:00"
url: https://arxiv.org/abs/2501.06530v1
source: arxiv
doi: null
arxiv_id: 2501.06530v1
categories:
  - cs.SD
  - eess.AS
---

# Multi-modal Speech Enhancement with Limited Electromyography Channels

Fuyuan Feng¹, Longting Xu¹ and Rohan Kumar Das² Affiliation: ¹College of
Information Science and Technology, Donghua University, Shanghai, China
Affiliation: ²Fortemedia Singapore, Singapore  
2232069@mail.edu.dhu.cn, xlt@dhu.edu.cn, rohankd@fortemedia.com

###### Abstract

Speech enhancement (SE) aims to improve the clarity, intelligibility,
and quality of speech signals for various speech enabled applications.
However, air-conducted (AC) speech is highly susceptible to ambient
noise, particularly in low signal-to-noise ratio (SNR) and
non-stationary noise environments. Incorporating multi-modal information
has shown promise in enhancing speech in such challenging scenarios.
Electromyography (EMG) signals, which capture muscle activity during
speech production, offer noise-resistant properties beneficial for SE in
adverse conditions. Most previous EMG-based SE methods required 35 EMG
channels, limiting their practicality. To address this, we propose a
novel method that considers only 8-channel EMG signals with acoustic
signals using a modified SEMamba network with added cross-modality
modules. Our experiments demonstrate substantial improvements in speech
quality and intelligibility over traditional approaches, especially in
extremely low SNR settings. Notably, compared to the SE (AC) approach,
our method achieves a significant PESQ gain of 0.235 under matched low
SNR conditions and 0.527 under mismatched conditions, highlighting its
robustness.

###### Index Terms: 

Electromyography, speech enhancement, multi-modal, Mamba

## I Introduction

Speech enhancement (SE) aims to improve the intelligibility and quality
of noisy speech signals, which is crucial for applications like speech
recognition and hearing aids. Traditional methods, such as spectral
subtraction, Wiener filtering and nonnegative matrix factorization (NMF)
\[[1](#bib.bib1)\] often struggle in complex noise environments. The
rise of deep learning has greatly advanced SE, with early models like
feedforward neural networks (FNNs), convolutional neural networks (CNNs)
\[[2](#bib.bib2), [3](#bib.bib3)\], and long short-term memory (LSTM)
networks \[[4](#bib.bib4), [5](#bib.bib5), [6](#bib.bib6)\] focusing on
the time-frequency domain to estimate clean speech. More recently,
end-to-end architectures \[[7](#bib.bib7)\] have been developed to
directly estimate clean speech in the time domain, achieving better
results. Advanced approaches, such as generative adversarial networks
(GANs) \[[8](#bib.bib8), [9](#bib.bib9), [10](#bib.bib10),
[11](#bib.bib11)\] and attention-based models, have further enhanced SE.

A recent promising development in SE is the Mamba architecture, which
leverages state-space models with a selection mechanism. Rong Chao et
al. \[[12](#bib.bib12)\] conducted a comparative study of
transformer-based and Mamba-based SE models and introduced a novel
system namely SEMamba. Their approach utilizes a bidirectional Mamba
architecture where the input is processed in parallel through the Mamba
network and subsequently connected to the output. This bidirectional
Mamba is applied concurrently in both the time and frequency domains, a
configuration referred to as TF-Mamba. Their studies demonstrate that
TF-Mamba shows significant potential for improving SE performance.

![Refer to caption](2501.06530v1/icassp.drawio.png)

Fig. 1: Architecture of multi-modal SE based on SU-E2S and modified
SEMamba.

All these mentioned SE methods were developed using air-conducted (AC)
speech. However, due to the nature of air conduction, AC speech is
highly susceptible to ambient noise, which significantly degrades its
performance in low SNR and non-stationary noise environments. For
example, security guards or on-site agents who perform tasks at famous
sports events will have a lot of noise, especially language
noise \[[13](#bib.bib13)\]. To overcome these limitations, alternative
modalities like video \[[14](#bib.bib14)\] have been explored to enhance
target speech.

However, audio-visual enhancement relies on camera-equipped devices,
limiting its use in outdoor, underwater, or military settings. Another
promising solution is the use of Electromyography (EMG) signals, which
can be non-invasively recorded through skin-attached electrodes. EMG
signals from the throat and face during speech provide valuable
speech-related information, and numerous studies have shown the
viability of EMG in speech applications.

Diener et al. \[[15](#bib.bib15)\] compute stacked time-domain features
from windowed EMG signals and map these features to parallel acoustic
features, subsequently, a vocoder synthesizes the acoustic waveform from
the acoustic feature predictions. Gaddy and Klein \[[16](#bib.bib16),
[17](#bib.bib17)\] process EMG signals with convolutional layers and a
Transformer encoder. They align EMG signals of silent articulation with
a reference audio by performing dynamic time warping (DTW)
\[[18](#bib.bib18)\]. Matthias Janke et al. \[[19](#bib.bib19)\]
proposed directly converting EMG signals into audible speech waveforms
and compared methods for extracting time-domain features from EMG
signals.

In \[[20](#bib.bib20)\], the authors proposed Speech-Unit-based
EMG-to-Speech (SU-E2S) modal, which predicts soft speech units from EMG
signals and uses a pre-trained acoustic VC decoder to reconstruct
acoustic features. Inspired by this article, SU-E2S modal was also
adopted in this work due to the fact that the EMG encoder of the SU-E2S
system is not trained to predict the original acoustic features, but
rather representations of the spoken content. Therefore, we hope to
learn richer semantic information independent of speaker
features \[[21](#bib.bib21)\]. Recently, the authors
of \[[22](#bib.bib22)\] applied EMG signals to SE for the first time,
using noisy spectrogram and 15 time-domain features (TD15 vectors)
extracted from 35 channels of EMG as inputs for SE, and achieved
improved results in some noisy scenarios.

However, previous EMG-based SE methods typically required 35 EMG
channels, which significantly limited their usage for practical
applications. In contrast, our approach is designed to use only 8 EMG
channels, making it more feasible for real-world scenarios. To achieve
this, we have adapted the latest SEMamba framework for multi-modal SE by
integrating additional input modalities to enhance the performance. Our
approach operates in two stages: In the first stage, we predict speech
signals from 8-channel EMG signals that are unaffected by environmental
noise. In the second stage, these predicted speech signals, together
with noisy speech signals, are used as inputs to our model to further
enhance the speech. This two-stage strategy contributes in both speech
quality and intelligibility compared to traditional noise-only methods,
especially in cases with extremely low SNR.

## II Multi-Modal SE: Processing AC Speech and EMG Signals with SEMamba

As illustrated in
Fig. [1](#S1.F1 "Fig. 1 ‣ I Introduction ‣ Multi-modal Speech Enhancement with Limited Electromyography Channels"),
the proposed multi-modal SE based on the modified SEMamba has two
stages. In the first stage, EMG signals recorded during voiced
pronunciation are converted into speech signals. Then, in the second
stage, the predicted speech from EMG and the noisy speech are taken as
input into our proposed multi-modal SE. The details of the two stages
are described in the following subsections.

### II-A Speech-Unit-based EMG-to-Speech (SU-E2S)

For the EMG encoder, we adopt the network architecture from Gaddy and
Klein \[[16](#bib.bib16)\], incorporating modifications specifically for
soft SU prediction. Initially, a convolutional layer downsamples the EMG
signal $`(\mathbf{X}_{1},\dots,\mathbf{X}_{T})`$ to a frame rate of
50Hz, aligning it with the frequency of the soft speech units. The
resulting feature sequence is then processed by a transformer encoder,
which outputs two predictions: the soft speech units
$`(\hat{c}_{1},\ldots,\hat{c}_{\hat{C}})`$ and the phonemes
$`(\hat{p}_{1},\ldots,\hat{p}_{\hat{C}})`$. The encoder is trained by
minimizing the distance between the predicted and target soft speech
units $`({c}_{1},\ldots,{c}_{{C}})`$, as well as by optimizing the
cross-entropy loss for phoneme classification, a method shown to enhance
performance in related studies.

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
\mathcal{L}_{SU}=\frac{1}{C}\sum_{t=1}^{C}\left\|\mathbf{c}_{t}-E_{c}(\mathbf{x})_{t}\right\|_{2}
``` |  | (1) |

where $`E_{c}(\mathbf{x})_{t}`$ denotes the speech units predictions of
the EMG encoder at frame $`t`$.

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{P}=-\frac{1}{C}\sum_{t=1}^{C}\sum_{i=1}^{|\mathcal{P}|}\mathbf{b}_{t,i}\cdot\log E_{p}(\mathbf{x})_{t,i}
``` |  | (2) |

where $`\mathbf{b}_{t,i}`$ is the binary indicator that phoneme $`i`$ is
the target class at frame $`t`$. $`E_{p}(\mathbf{x})_{t}`$ is a sequence
of probability distributions for phoneme predictions, with a length of
$`C`$. The total loss is a weighted sum of $`\mathcal{L}_{SU}`$ and
$`\mathcal{L}_{P}`$:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{total}}=\lambda_{SU}\mathcal{L}_{SU}+\lambda_{P}\mathcal{L}_{P}
``` |  | (3) |

Where $`\lambda_{SU}`$ and $`\lambda_{P}`$ are scalar weights for the
respective components.

After obtaining the predicted soft speech units \[[20](#bib.bib20)\], we
use a pre-trained acoustic decoder with a convolutional module
\[[23](#bib.bib23)\], a Pre-Net, and an autoregressive LSTM to convert
predicted speech units into Mel-spectrograms. The network’s
encoder-decoder structure transforms discrete or soft speech units into
spectrograms for efficient voice conversion. Subsequently, just like the
direct EMG-to-Speech model \[[15](#bib.bib15), [24](#bib.bib24)\], a
vocoder synthesizes the acoustic signal from the predicted features.
Here, we use a pre-trained HiFi-GAN model \[[25](#bib.bib25)\] to
perform this synthesis, ensuring high-quality audio generation from the
predicted Mel spectrograms.

This is the process of the first stage, highlighted in the blue box at
the top of
Fig. [1](#S1.F1 "Fig. 1 ‣ I Introduction ‣ Multi-modal Speech Enhancement with Limited Electromyography Channels"),
involves converting the EMG signal into an EMG-predicted speech signal
that is robust to environmental noise by predicting soft speech units.

### II-B Multi-modal SE based on modified SEMamba

Our implementation of multi-modal SE is based on SEMamba
\[[12](#bib.bib12)\]. In environments with excessive noise, enhancing
the noisy speech signals alone may not produce satisfactory results
\[[14](#bib.bib14), [26](#bib.bib26)\]. Therefore, we use both the
EMG-predicted speech, obtained as described in the previous subsection,
and the noisy speech as inputs. These two signals are synchronized and
correspond to the same clean speech.

As depicted in the purple box in the lower half of
Fig. [1](#S1.F1 "Fig. 1 ‣ I Introduction ‣ Multi-modal Speech Enhancement with Limited Electromyography Channels"),
the process begins with applying short-time Fourier transform (STFT) to
both input signals to obtain their spectral representations. The
magnitude component is then compressed and stacked with the phase
component. These stacked components are fed into a feature encoder,
which performs initial feature extraction on each of the speech signals
separately. The feature encoder uses a DenseEncoder architecture with a
DenseNet core featuring dilated convolutions, complemented by standard
convolutional layers on either end for multi-scale feature extraction.
Following this initial extraction, a cross module comprising fully
connected layers is employed to fuse the features from both signals,
effectively integrating rich speech information from each source. The
fused output is then processed by the TF-Mamba module, which extracts
deeper time-frequency domain characteristics.

TF-Mamba consists of two bidirectional Mamba blocks operating in both
the time and frequency domains. The input is processed in parallel
through the Mamba network, and the outputs are concatenated. This
combined output is then fed into a Conv1D layer, formulated as:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{y}=\textit{Conv1D}(M_{\mathit{uni}}(\mathbf{x})\oplus M_{\mathit{uni}}(\mathit{flip}(\mathbf{x})))
``` |  | (4) |

where $`\mathbf{x}`$, $`\mathbf{y}`$, $`M_{\mathit{uni}}()`$,
$`\mathit{flip}()`$, Conv1D(), and $`\oplus`$ represent the input,
output, uni-directional Mamba operation, flipping operation, 1-D
convolution, and concatenation, respectively.

The output from TF-Mamba is then fed into two separate decoders: one
reconstructs the magnitude mask, while the other reconstructs the real
and imaginary parts of the waveform. Both decoders use DenseBlock
structures with dilated convolutions and convolutional layers to
facilitate feature extraction as well as reconstruction. Finally, the
enhanced speech is obtained by applying the inverse short-time Fourier
transform (iSTFT).

## III Experiments

### III-A Dataset

Our study focuses on speech and EMG signals in scenarios involving
audible speech production. Therefore, we evaluate our model on the
corpus from Gaddy and Klein \[[16](#bib.bib16)\], which documents
subjects reading English sentences under both audible and silent
articulation conditions. The EMG signals in this corpus are recorded in
8-channel with a sampling rate of 1000Hz, while the audio are recorded
in 16 kHz. We use the same EMG filtering steps as in the authors’
implementation in \[[20](#bib.bib20)\]. We use the predefined validation
and testing splits, but use utterances with EMG signals of audible
articulation. Then the train, validation, and test splits contain 6755,
199, and 98 utterances, respectively. For the training and validation
sets, we applied MUSAN \[[27](#bib.bib27)\] to generate noisy audio
data. Each utterance is corrupted with five randomly selected types of
noise at five SNRs (-10, -5, 0, 5, and 10 dB). For the test set, we used
18 unseen noise types (car noise, engine noise, pink noise, white noise,
two types of street noises, six background Chinese speakers, and six
English speakers) to clean utterances at four SNRs (-11, -6, -1, and 4
dB) to create mismatch conditions \[[22](#bib.bib22)\]. It is noted that
background speakers in different languages are used to create babble
noise in language-specific conditions.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | SNR | Noisy speech |  |  | SE (AC) |  |  | Proposed (TF-Mamba$`\times`$1) |  |  | Proposed (TF-Mamba$`\times`$4) |  |  | Proposed (TF-Mamba$`\times`$8) |  |
|  |  | PESQ | STOI |  | PESQ | STOI |  | PESQ | STOI |  | PESQ | STOI |  | PESQ | STOI |
| Match | -10db | 1.304 | 0.682 |  | 2.641 | 0.898 |  | 2.672 | 0.905 |  | 2.876 | 0.925 |  | 2.778 | 0.917 |
|  | -5db | 1.26 | 0.75 |  | 2.945 | 0.929 |  | 2.9 | 0.928 |  | 3.094 | 0.943 |  | 3.024 | 0.937 |
|  | 0db | 1.365 | 0.831 |  | 3.327 | 0.954 |  | 3.24 | 0.95 |  | 3.404 | 0.961 |  | 3.363 | 0.956 |
|  | 5db | 1.54 | 0.89 |  | 3.624 | 0.971 |  | 3.519 | 0.967 |  | 3.655 | 0.974 |  | 3.624 | 0.97 |
|  | average | 1.367 | 0.788 |  | 3.134 | 0.938 |  | 3.083 | 0.938 |  | 3.257 | 0.951 |  | 3.197 | 0.945 |
| Mismatch | -11db | 1.174 | 0.533 |  | 1.554 | 0.756 |  | 1.917 | 0.835 |  | 2.081 | 0.862 |  | 1.972 | 0.851 |
|  | -6db | 1.139 | 0.637 |  | 1.885 | 0.846 |  | 2.236 | 0.878 |  | 2.444 | 0.903 |  | 2.353 | 0.893 |
|  | -1db | 1.176 | 0.738 |  | 2.263 | 0.901 |  | 2.544 | 0.91 |  | 2.774 | 0.931 |  | 2.712 | 0.923 |
|  | 4db | 1.262 | 0.831 |  | 2.77 | 0.935 |  | 2.91 | 0.938 |  | 3.126 | 0.954 |  | 3.083 | 0.947 |
|  | average | 1.188 | 0.685 |  | 2.118 | 0.86 |  | 2.402 | 0.89 |  | 2.606 | 0.913 |  | 2.53 | 0.904 |

TABLE I: Performance of SE (AC) and proposed method with different
number of TF-Mamba blocks under different SNR. The matched condition
uses 5 noises from MUSAN at 4 SNRs (-10, -5, 0, 5 dB) similar to
training, while the mismatched condition uses 18 unseen noises at SNRs
(-11, -6, -1, 4 dB).

### III-B Reference baselines and proposed method

In our experiments, we compared the performance of our proposed method
against several baselines to evaluate its effectiveness in different
noise environments, which are:

- •
  Noisy Speech: This baseline represents the raw, unprocessed noisy
  speech. It serves as the starting point for quality and
  intelligibility without any enhancement.
- •
  SE (AC): It has the same network structure to that of
  TF-Mamba$`\times`$4, with the difference being that the input only
  considers noisy speech but no EMG signals, thus developed for
  uni-modal SE.
- •
  Proposed: It is the multi-modal SE based on modified SEMamba. We
  tested with different numbers of TF-Mamba blocks, including 1, 4, and
  8 blocks, referred to as TF-Mamba$`\times`$1, TF-Mamba$`\times`$4, and
  TF-Mamba$`\times`$8, respectively. Each configuration utilizes both
  EMG and noisy speech to improve the quality and comprehensibility of
  speech.

### III-C Implementation details and evaluation metrics

In the first stage, we train the EMG encoder by minimizing the loss
function $`\mathcal{L}_{\text{total}}`$ with weights $`\lambda_{SU}`$ =
0.5 and $`\lambda_{P}`$ = 0.5, and the learning rate is set to 0.0003.
For the pre-trained acoustic decoder, we use a learning rate of 0.0001
and learn for 80k steps. The loss function in the second stage is a
linear combination of the PESQ-based GAN discriminator loss, and losses
based on time, magnitude, complex, and phase
components \[[28](#bib.bib28)\].

In this study, we use the perceptual evaluation of speech quality
(PESQ) \[[29](#bib.bib29)\] as an objective quality measure to assess
the quality of the speech signals. In addition, short-time objective
intelligibility (STOI) \[[30](#bib.bib30)\] is used for intelligibility
assessment of the speech signals.

## IV Results and Discussion

Table [I](#S3.T1 "TABLE I ‣ III-A Dataset ‣ III Experiments ‣ Multi-modal Speech Enhancement with Limited Electromyography Channels")
shows the performance comparison of our proposed method to the reference
baselines under various conditions. We define the matched condition as
the test set comprising 5 noises randomly selected from
MUSAN \[[27](#bib.bib27)\] at 4 SNRs (-10, -5, 0, and 5 dB), consistent
with the types of noise encountered during training. In contrast, the
mismatched condition uses a test set of 5 noises randomly selected from
18 unseen noises at SNRs of -11, -6, -1, and 4 dB. We discuss the
results of various studies conducted in the following subsections.

#### IV-1 Matched low SNR conditions

Under matched conditions (-10 dB and -5 dB), TF-Mamba$`\times`$4
demonstrates substantial improvements over the baseline SE (AC) method.
At -10dB, TF-Mamba$`\times`$4 achieves a PESQ score of 2.876, compared
to SE (AC)’s 2.641, marking an improvement of 0.235. Similarly, at -5dB,
TF-Mamba$`\times`$4 obtains a PESQ of 3.094, exceeding SE (AC) by 0.149.
These gains are particularly significant given the difficult nature of
these low SNR environments. The STOI scores also show a similar pattern,
with TF-Mamba$`\times`$4 achieving 0.925 at -10dB and 0.943 at -5dB,
indicating clear enhancements in speech intelligibility.

#### IV-2 Mismatched low SNR conditions

In mismatched conditions (-11 dB and -6 dB), the benefits of
TF-Mamba$`\times`$4 are even more pronounced. At -11dB,
TF-Mamba$`\times`$4 achieves a PESQ of 2.081, significantly higher than
SE (AC)’s 1.554, representing an increase of 0.527. This enhancement
persists at -6 dB, where TF-Mamba$`\times`$4 reaches a PESQ score of
2.444, outperforming the baseline’s 1.885 by 0.559. The STOI
improvements are similarly notable, with TF-Mamba$`\times`$4 achieving
0.862 at -11dB and 0.903 at -6dB, reflecting substantial gains in
intelligibility under these extreme conditions.

#### IV-3 Ablation study with different number of TF-Mamba blocks

Table [I](#S3.T1 "TABLE I ‣ III-A Dataset ‣ III Experiments ‣ Multi-modal Speech Enhancement with Limited Electromyography Channels")
also demonstrate that increasing the number of TF-Mamba blocks from 1 to
4 leads to significant improvements in performance, while further
increasing to 8 blocks results in only marginal gains, indicating
diminishing returns. The minimal difference between TF-Mamba$`\times`$4
and TF-Mamba$`\times`$8 suggests that four blocks strike an optimal
balance between performance and computational efficiency.

We are now interested to observe the impact of proposed method under
different types of noise.
Table [II](#S4.T2 "TABLE II ‣ IV-3 Ablation study with different number of TF-Mamba blocks ‣ IV Results and Discussion ‣ Multi-modal Speech Enhancement with Limited Electromyography Channels")
highlights the effectiveness of our TF-Mamba$`\times`$4 method compared
to the baseline noisy speech and SE (AC) approaches across various noise
types. Our method consistently outperforms the SE (AC) in both PESQ and
STOI scores. Notably, in complex noise environments like pink and white
noise, it shows greater improvements in speech quality and
intelligibility. Again, considering the babble noise in English and
Chinese speaking speaker environments, the relative performance
improvement was more significant in English babble noise environments.

|         |              |       |     |         |       |     |          |       |
|---------|--------------|-------|-----|---------|-------|-----|----------|-------|
|         | Noisy speech |       |     | SE (AC) |       |     | Proposed |       |
|         | PESQ         | STOI  |     | PESQ    | STOI  |     | PESQ     | STOI  |
| Car     | 1.117        | 0.762 |     | 2.351   | 0.903 |     | 2.588    | 0.926 |
| Engine  | 1.086        | 0.645 |     | 1.821   | 0.793 |     | 2.277    | 0.873 |
| Pink    | 1.056        | 0.69  |     | 1.73    | 0.805 |     | 2.233    | 0.881 |
| White   | 1.051        | 0.729 |     | 1.785   | 0.832 |     | 2.281    | 0.894 |
| Street  | 1.145        | 0.673 |     | 1.979   | 0.83  |     | 2.386    | 0.891 |
| English | 1.213        | 0.665 |     | 2.121   | 0.869 |     | 2.707    | 0.92  |
| Chinese | 1.247        | 0.694 |     | 2.312   | 0.878 |     | 2.758    | 0.925 |

TABLE II: Performance of SE (AC) and proposed method
(TF-Mamba$`\times`$4) for different noise types.

These results discussed above for various noise conditions and types
demonstrate that our proposed TF-Mamba$`\times`$4 model effectively
integrates multi-modal information from EMG and speech, leading to
substantial improvements in speech quality and intelligibility in
challenging and language-specific babble noise conditions. It is worth
to be noted that this was achieved by considering only 8 channels of EMG
signals.

## V Conclusion

In this study, we proposed a modified SEMamba framework for multi-modal
SE using 8-channel EMG signals combined with noisy speech. This approach
is evaluated in challenging environments with low SNR and
language-specific babble noise conditions. The integration of EMG with
noisy speech outperformed the uni-modal method that only considers noisy
speech, showing a significant improvement in PESQ and STOI. Our analysis
suggest that using four TF-Mamba blocks provides an optimal balance
between performance and computational efficiency, with limited gains
beyond this point. Notably, our method employs only 8 EMG channels, the
lowest count in multi-modal speech enhancement studies considering EMG
signals, underscoring its efficiency. However, there are challenges
remain in effectively fusing features from different modalities, which
may limit enhancement performance. Future research will focus on
developing advanced feature fusion techniques to fully leverage the
complementary information from EMG and acoustic signals, aiming to
enhance SE performance in diverse real-world environments.

## References

- \[1\] L. Xu, Z. Wei, S. F. A. Zaidi, B. Ren, and J. Yang, “Speech
  enhancement based on nonnegative matrix factorization in constant-q
  frequency domain,” *Applied Acoustics*, vol. 174, p. 107732, 2021.
- \[2\] S.-W. Fu, Y. Tsao, X. Lu *et al.*, “SNR-aware convolutional
  neural network modeling for speech enhancement.” in *Interspeech*,
  2016, pp. 3768–3772.
- \[3\] A. Li, M. Yuan, C. Zheng, and X. Li, “Speech enhancement using
  progressive learning-based convolutional recurrent neural network,”
  *Applied Acoustics*, vol. 166, p. 107347, 2020.
- \[4\] F. Weninger, H. Erdogan, S. Watanabe, E. Vincent, J. Le Roux,
  J. R. Hershey, and B. Schuller, “Speech enhancement with LSTM
  recurrent neural networks and its application to noise-robust ASR,” in
  *International Conference on Latent Variable Analysis and Signal
  Separation*. Springer, 2015, pp. 91–99.
- \[5\] R. Liang, F. Kong, Y. Xie, G. Tang, and J. Cheng, “Real-time
  speech enhancement algorithm based on attention LSTM,” *IEEE Access*,
  vol. 8, pp. 48 464–48 476, 2020.
- \[6\] Z. Chen, S. Watanabe, H. Erdogan, and J. Hershey, “Integration
  of speech enhancement and recognition using long-short term memory
  recurrent neural network,” in *Interspeech*, 2015, pp. 1–7.
- \[7\] S.-W. Fu, T.-W. Wang, Y. Tsao, X. Lu, and H. Kawai, “End-to-end
  waveform utterance enhancement for direct evaluation metrics
  optimization by fully convolutional neural networks,” *IEEE/ACM
  Transactions on Audio, Speech, and Language Processing*, vol. 26,
  no. 9, pp. 1570–1584, 2018.
- \[8\] S. Abdulatif, R. Cao, and B. Yang, “CMGAN: Conformer-based
  metric-GAN for monaural speech enhancement,” *IEEE/ACM Transactions on
  Audio, Speech, and Language Processing*, vol. 32, pp. 2477 –
  2493, 2024.
- \[9\] C. Donahue, B. Li, and R. Prabhavalkar, “Exploring speech
  enhancement with generative adversarial networks for robust speech
  recognition,” in *IEEE International Conference on Acoustics, Speech,
  and Signal Processing (ICASSP)*, 2018, pp. 5024–5028.
- \[10\] H. Phan, I. V. McLoughlin, L. Pham, O. Y. Chén, P. Koch,
  M. De Vos, and A. Mertins, “Improving GANs for speech enhancement,”
  *IEEE Signal Processing Letters*, vol. 27, pp. 1700–1704, 2020.
- \[11\] Y. Ji, L. Xu, and W.-P. Zhu, “Adversarial dictionary learning
  for monaural speech enhancement.” in *Interspeech*, 2020, pp.
  4034–4038.
- \[12\] R. Chao, W.-H. Cheng, M. La Quatra, S. M. Siniscalchi, C.-H. H.
  Yang, S.-W. Fu, and Y. Tsao, “An investigation of incorporating Mamba
  for speech enhancement,” *arXiv preprint arXiv:2405.06573*, 2024.
- \[13\] H. Taherian, Z.-Q. Wang, J. Chang, and D. Wang, “Robust speaker
  recognition based on single-channel and multi-channel speech
  enhancement,” *IEEE/ACM Transactions on Audio, Speech, and Language
  Processing*, vol. 28, pp. 1293–1302, 2020.
- \[14\] J.-C. Hou, S.-S. Wang, Y.-H. Lai, Y. Tsao, H.-W. Chang, and
  H.-M. Wang, “Audio-visual speech enhancement using multimodal deep
  convolutional neural networks,” *IEEE Transactions on Emerging Topics
  in Computational Intelligence*, vol. 2, no. 2, pp. 117–128, 2018.
- \[15\] L. Diener, C. Herff, M. Janke, and T. Schultz, “An initial
  investigation into the real-time conversion of facial surface EMG
  signals to audible speech,” in *International Conference of the IEEE
  Engineering in Medicine and Biology Society (EMBC)*, 2016, pp.
  888–891.
- \[16\] D. Gaddy and D. Klein, “Digital voicing of silent speech,” in
  *Empirical Methods in Natural Language Processing (EMNLP)*, 2020, pp.
  5521–5530.
- \[17\] J. A. Gonzalez, L. A. Cheah, J. M. Gilbert, J. Bai, S. R. Ell,
  P. D. Green, and R. K. Moore, “A silent speech system based on
  permanent magnet articulography and direct synthesis,” *Computer
  Speech & Language*, vol. 39, pp. 67–87, 2016.
- \[18\] L. Muda, “Voice recognition algorithms using mel frequency
  cepstral coefficient (MFCC) and dynamic time warping (DTW)
  techniques,” *arXiv preprint arXiv:1003.4083*, 2010.
- \[19\] M. Janke and L. Diener, “EMG-to-speech: Direct generation of
  speech from facial electromyographic signals,” *IEEE/ACM Transactions
  on Audio, Speech, and Language Processing*, vol. 25, no. 12, pp.
  2375–2385, 2017.
- \[20\] K. Scheck and T. Schultz, “Multi-speaker speech synthesis from
  electromyographic signals by soft speech unit prediction,” in *IEEE
  International Conference on Acoustics, Speech and Signal Processing
  (ICASSP)*, 2023, pp. 1–5.
- \[21\] K. Scheck, Z. Ren, T. Dombeck, J. Sonnert, S. van Gogh, Q. Hou,
  M. Wand, and T. Schultz, “Cross-speaker training and adaptation for
  electromyography-to-speech conversion,” in *International Conference
  of the IEEE Engineering in Medicine and Biology Society (EMBC)*, 2024,
  pp. 1–4.
- \[22\] K.-C. Wang, K.-C. Liu, H.-M. Wang, and Y. Tsao, “EMGSE:
  Acoustic/EMG fusion for multimodal speech enhancement,” in *IEEE
  International Conference on Acoustics, Speech and Signal Processing
  (ICASSP)*, 2022, pp. 1116–1120.
- \[23\] B. Van Niekerk, M.-A. Carbonneau, J. Zaïdi, M. Baas, H. Seuté,
  and H. Kamper, “A comparison of discrete and soft speech units for
  improved voice conversion,” in *IEEE International Conference on
  Acoustics, Speech and Signal Processing (ICASSP)*, 2022, pp.
  6562–6566.
- \[24\] K. Scheck, D. Ivucic, Z. Ren, and T. Schultz, “Stream-ETS:
  Low-latency end-to-end speech synthesis from electromyography
  signals,” in *ITG Conference on Speech Communication*, 2023, pp.
  200–204.
- \[25\] J. Kong, J. Kim, and J. Bae, “HiFi-GAN: Generative adversarial
  networks for efficient and high fidelity speech synthesis,” *Advances
  in Neural Information Processing Systems*, vol. 33, pp.
  17 022–17 033, 2020.
- \[26\] M. Tagliasacchi, Y. Li, K. Misiunas, and D. Roblek, “SEANet: A
  multi-modal speech enhancement network,” in *Interspeech*, 2020, pp.
  1126–1130.
- \[27\] D. Snyder, G. Chen, and D. Povey, “MUSAN: A music, speech, and
  noise corpus,” *arXiv preprint arXiv:1510.08484*, 2015.
- \[28\] Y.-X. Lu, Y. Ai, and Z.-H. Ling, “MP-SENet: A speech
  enhancement model with parallel denoising of magnitude and phase
  spectra,” in *Interspeech*, 2023, pp. 3834–3838.
- \[29\] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra,
  “Perceptual evaluation of speech quality (PESQ)-a new method for
  speech quality assessment of telephone networks and codecs,” in *IEEE
  International Conference on Acoustics, Speech, and Signal Processing
  (ICASSP)*, 2001, pp. 749–752.
- \[30\] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “An
  algorithm for intelligibility prediction of time–frequency weighted
  noisy speech,” *IEEE Transactions on Audio, Speech, and Language
  Processing*, vol. 19, no. 7, pp. 2125–2136, 2011.
````
