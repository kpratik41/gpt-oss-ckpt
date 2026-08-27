# Watermarking LLM Text Output with SynthID-Text

**An evaluation and a reusable toolkit. The method is model-agnostic; it was tested here on
Gemma-4.**

Pratik Kakkar · Chandra Dhir · Anup Shirgaonkar

Models `google/gemma-4-E4B-it` and `google/gemma-4-31B-it` · fluency judged by
`Qwen/Qwen3.8-27B` · 2 × NVIDIA RTX PRO 6000 Blackwell (97 GB) · SynthID-Text via
Hugging Face Transformers 5.12.1 · package `synthmark` v0.1.0

---

## 1. Why this exists

### The legal driver

The EU AI Act (Regulation 2024/1689) Article 50(2) requires providers of generative AI to
ensure outputs are **"marked in a machine-readable format and detectable as artificially
generated or manipulated."** Article 50(4) separately requires *deployers* to disclose when
AI-generated text is published to inform the public on matters of public interest — with an
exemption where the text underwent human editorial review and a person holds editorial
responsibility.

Three details make this operationally urgent rather than theoretical:

- **The obligation is already live.** Article 50(4) applied from 2 August 2026. The
  machine-readable marking duty under 50(2) carries an AI Omnibus grace period to
  **2 December 2026** for systems already on the market before August.
- **It reaches outside the EU.** The Act applies extraterritorially to providers whose
  outputs reach EU users, regardless of where the firm is headquartered.
- **Penalties are turnover-scaled** — up to €15 million or 3% of worldwide annual turnover.

The Code of Practice on Transparency of AI-Generated Content is a Commission-confirmed route
to demonstrating compliance. Note that Article 50(2) requires marking to be effective *"as far
as this is technically feasible"* — the statute anticipates limits, so measuring them honestly
is part of a compliance argument rather than a weakness in one.

### What watermarking actually helps with

The wider harm picture is real. The FTC's Consumer Sentinel Network logged over 330,000
fraud reports involving AI-generated content or AI-assisted social engineering in 2025, and
the economics have collapsed: attacks that once needed a team and tens of thousands of
dollars now cost tens of dollars and under an hour. The Arup case — deepfaked
video-conference participants authorising $25 million in transfers — is the canonical
example of what synthetic media enables against a large organisation.

**Scope: this report studies text watermarking only.** Nothing here was tested on, or
applies to, audio, image or video output — those need different controls entirely (C2PA
content credentials for files, audio watermarking for speech). Most of the incidents above in
fact involve voice and video, which this work does not address.

Within text, the control is also narrower than it first appears: §6 shows a single paraphrase
pass removes the mark. It is not an anti-fraud control and will not stop a motivated
attacker.

What it does deliver:

| Genuinely helps | Does not help |
|---|---|
| Regulatory provenance — machine-readable marking under Art. 50(2) | Determined adversaries — one paraphrase pass defeats it (§6) |
| Distinguishing model output from human writing in unmodified reuse | Attribution — it identifies a *key*, never a person |
| Keeping model output out of training corpora (model-collapse hygiene) | Short text, code, structured output (§5.1) |
| Internal audit: was this filing, memo or client note machine-drafted? | Proving a human *didn't* write something — a negative is not evidence |
| Per-desk keys so one unit's detector cannot read another's traffic (§5.3) | Content the pipeline did not generate |

The realistic framing is **provenance for honest use, not enforcement against dishonest
use.** It answers "did our own systems produce this?" reliably, and "did someone
adversarially pass off AI text as human?" not at all.

§8 maps these findings to the compliance obligations in detail.

---

## 2. What was done

Two deliverables.

**A package (`synthmark`).** A thin, reusable layer over the SynthID-Text implementation
that ships in Hugging Face Transformers, adding the parts a real deployment needs and the
reference code omits: key derivation and storage, calibrated decision thresholds, a
CPU-only detection service, a CLI, and an evaluation harness. It is model-agnostic; Gemma-4
is the model it was validated on.

**An evaluation.** A measurement of what the watermark actually delivers on this model —
how detectable it is, how much text it needs, whether it degrades output, what destroys it,
and what it costs — with the limitations stated rather than glossed.

One substantive bug was found and fixed along the way; see §5.

---

## 3. How the watermark works

### Two schemes ship with Transformers. This is the one we evaluated, and why

Transformers ships **two unrelated watermarking schemes**, and they behave very differently.
Everything in this report concerns the second approach.

| | Green-list (Kirchenbauer et al.) | SynthID-Text (used here) |
|---|---|---|
| Logits processor | `WatermarkLogitsProcessor` | `SynthIDTextWatermarkLogitsProcessor` |
| Config | `WatermarkingConfig` | `SynthIDTextWatermarkingConfig` |
| Detector | `WatermarkDetector` (z-score on green-token count) | `SynthIDTextWatermarkDetector`, `BayesianDetectorModel` |
| Mechanism | Splits the vocabulary into "green" and "red" each step, adds a constant `bias` to green logits | Tournament among candidates, reweighting by g-value |
| Distortionary? | **Yes** | **No, by design** |
| Source | [arXiv 2306.04634](https://huggingface.co/papers/2306.04634) | [Nature 634 (2024)](https://www.nature.com/articles/s41586-024-08025-4) |

The green-list scheme partitions the vocabulary into green and red tokens at each step and
adds a constant `bias` (default 2.0, over a green fraction of 0.25) to the green logits. **That
does push the model towards tokens it would not otherwise have picked: it is distortionary,
and it trades quality for detectability through that bias knob.** Upstream says
so itself — the parameter's own documentation reads *"Consider lowering the `bias` if the
text generation quality degrades."*

SynthID's tournament is built the other way round. **It reweights *among* the candidates the
model already found plausible while preserving total probability mass, so a token the model
gave near-zero probability stays near zero.** That is precisely why §5.5 finds no quality cost,
and why there is no quality/detectability dial to tune.

**Why we chose SynthID.** The requirement driving this work (§1) is **machine-readable
provenance that does not degrade a production service**. Only a non-distortionary scheme can
satisfy both halves of that: the green-list approach makes quality the currency it pays for
detectability with, so any deployment of it must argue about where to set `bias`, and that
argument has no good answer for a bank shipping client-facing text. SynthID removes the
question. It is also the scheme behind the EU Code of Practice signatories' deployments and
the one with a published, peer-reviewed evaluation at scale.

**Scope of the headline result: "no quality degradation" is specific to SynthID and does not
transfer to the green-list approach.** Anyone benchmarking the
other scheme should expect a real quality/detectability trade-off, and should not cite these
numbers in support of it.

A naming trap worth flagging: the *unprefixed* classes — `WatermarkingConfig`,
`WatermarkDetector` — are the green-list scheme. `WatermarkDetector` will accept
SynthID-generated text without error and report nothing, which looks identical to genuinely
unwatermarked text. `synthmark` imports only the `SynthID*` classes and never reaches the
green-list path.

---

At each decoding step the model produces a distribution over next tokens. Where several
tokens are near-equally good — after *"the weather was cold and…"*, both *"grey"* and
*"overcast"* work — an ordinary sampler picks between them with a random number. **SynthID
replaces that random number with a pseudorandom function** ("g-function") seeded by a secret
key and a sliding window of the preceding tokens, then runs a small tournament among the
candidates that biases the outcome towards tokens with high g-values.

Nothing is inserted into the text: no hidden characters, no markup, no extra tokens. The
signal is *which* of the equally-good tokens was chosen. Detection replays the same key
over the candidate text, computes the g-values that generation would have used, and asks
whether they are biased upward. Under the null hypothesis — text not watermarked with this
key — each g-value is an unbiased coin flip and the mean sits at 0.5.

### What that looks like on real text

Here is a passage the model actually generated, token by token. Each token gets 30
independent coin flips — one per key in the bundle — seeded by the secret key and the four
tokens immediately before it. The same passage is also scored under a *different* key.

| # | Token | Heads (of 30) | g | Running mean g | g, wrong key | Running mean, wrong key |
|---|---|---|---|---|---|---|
| 1 | ` relentless` | 17/30 | 0.567 | **0.567** | 0.567 | 0.567 |
| 2 | ` sculptor` | 13/30 | 0.433 | **0.500** | 0.633 | 0.600 |
| 3 | ` against` | 19/30 | 0.633 | **0.544** | 0.700 | 0.633 |
| 4 | ` the` | 16/30 | 0.533 | **0.542** | 0.567 | 0.617 |
| 5 | ` granite` | 15/30 | 0.500 | **0.533** | 0.467 | 0.587 |
| 6 | ` tower` | 13/30 | 0.433 | **0.517** | 0.400 | 0.556 |
| 7 | ` of` | 14/30 | 0.467 | **0.510** | 0.500 | 0.548 |
| 8 | ` the` | 11/30 | 0.367 | **0.492** | 0.567 | 0.550 |
| 9 | ` Black` | 16/30 | 0.533 | **0.496** | 0.500 | 0.544 |
| 10 | `water` | 17/30 | 0.567 | **0.503** | 0.467 | 0.537 |
| 11 | ` Point` | 14/30 | 0.467 | **0.500** | 0.600 | 0.542 |
| 12 | ` lighthouse` | 15/30 | 0.500 | **0.500** | 0.500 | 0.539 |
| 13 | `.` | 18/30 | 0.600 | **0.508** | 0.467 | 0.533 |
| 14 | ` Silas` | 19/30 | 0.633 | **0.517** | 0.600 | 0.538 |

Over the full 396-token passage: correct key **0.5372**, a different key 0.5096, null expectation 0.5000 (z = +8.11, p = 2.5e-16).

This is the whole idea in one table. **No individual token is a tell.** Token 8 comes up
0.367 under the correct key, well *below* chance; token 3 comes up 0.633 under the wrong
one. Reading any single row, you could not say which key produced the text — which is
precisely why a reader sees nothing unusual and why the watermark costs no quality.

**The signal lives only in the average.** Watch the two running means: they are
indistinguishable for the first dozen tokens, and only over hundreds of tokens does the
correct key's average settle above 0.5 while the wrong key's stays at chance. That single
observation explains most of §5 — why detection needs length, why one key cannot read
another's traffic, and why an edit that changes tokens costs signal.

Two properties follow directly from the mechanism, and they bound everything in §5:

- **It can only use randomness that already exists.** Where the model is nearly certain of
  the next token — arithmetic, code, fixed schemas, terse factual answers — there is no
  choice to steer and little or no watermark is embedded.
- **It cannot make the output worse.** It never introduces a token the model would not
  otherwise have sampled. Quality is preserved *at the expense of* watermark strength, not
  the other way around.

## 4. Experimental setup

**Corpus.** 24 prompts × 4 independent samples across six prompt suites, generated twice —
once watermarked, once not — with the *same prompts and the same seeds* under both
conditions, so every comparison is paired. 400 new tokens per generation, sampled at
temperature 1.0, top-k 64, top-p 0.95 (the model's own defaults). 1,088 texts per model.

**Entropy, and why it governs everything.** At each decoding step the model produces a
probability distribution over all 262,144 tokens. *Entropy* measures how spread out that
distribution is:

- **High entropy** — after *"the weather was cold and…"*, perhaps twenty tokens are all
  plausible (`grey`, `overcast`, `damp`, `windy`). The sampler faces a genuine choice and
  must resolve it with a random draw.
- **Low entropy** — after *"2 + 2 ="*, essentially one token carries all the mass. There is
  no choice to make.

SynthID works by replacing the sampler's random draw with a key-derived one, so it can only
encode signal where a draw actually happens. Where the model is certain, the watermark has
nothing to steer and embeds nothing — however long the text runs. This single fact predicts
most of §5: which prompt suites are detectable, why length helps only up to a point, and why
a stronger model is harder to watermark.

**Reading AUC.** Detection results are reported as AUC and as TPR at a fixed false-positive
rate. AUC is the probability that a randomly chosen watermarked text scores higher than a
randomly chosen unwatermarked one: 0.5 is chance, 1.0 means the two score distributions do
not overlap at all.

AUC alone is the wrong number to govern a deployment with, because it is **threshold-free** —
it averages over every operating point, including ones with a false-positive rate nobody
would accept. The `code` suite makes the gap concrete: AUC 0.77 sounds workable, but at a
threshold that wrongly flags only 1 document in 100, it catches **1%** of watermarked code.
TPR at a fixed FPR answers the question that actually arises — *at a false-accusation rate we
can defend, what fraction do we catch?* — and it is the number quoted whenever a decision
depends on it.

**Two models.** Everything is measured on both `gemma-4-E4B-it` (~4B effective) and
`gemma-4-31B-it`. They share a tokenizer and a 262,144-token vocabulary, so comparing them
isolates the effect of model size. Robustness and the learned-detector comparison are
reported on E4B only; those measure how a *given* watermark signal degrades, and the E4B
corpus provides the strongest starting signal to degrade from.

**Fluency judge.** `Qwen/Qwen3.8-27B` — an unrelated architecture with its own tokenizer and
training data. A model cannot score its own output for this purpose (see §5.5), and a judge
from the same family would share the generator's idiosyncrasies.

The suites are split by **entropy**, because entropy is the variable that governs watermark
strength. Reporting a single aggregate number over a mixed prompt set would hide the effect
that matters most in practice.

| Suite | Content | Expected freedom in token choice |
|---|---|---|
| `creative` | Stories, descriptive writing | Highest |
| `open_ended` | Explanatory prose | High |
| `financial` | Client notes, product explanations | High |
| `factual` | Short answers pinned by facts | Low |
| `structured` | JSON with a fixed schema | Very low |
| `code` | Python, SQL, bash | Lowest |

**Key.** Watermarking depth 30, n-gram length 5 — the reference configuration from the
SynthID-Text paper. Keys are derived by HKDF-SHA256 from a single master secret, so a label
like `markets-research/v1` deterministically yields an independent key.

**Negative controls.** Three distinct kinds, because they answer different questions:

1. *Unwatermarked output from the same model* — does the detector key on the watermark, or
   merely on the model's style?
2. *Human-written text* (600 WikiText-103 passages, detokenised) — how often would we
   wrongly flag a person's writing?
3. *The same watermarked text scored with a different key* — can one desk's detector read
   another desk's traffic?

---

## 5. Results

### 5.0 Summary of claims

| Claim | Verdict | Evidence |
|---|---|---|
| The watermark is detectable with our key | ⚠️ | **Model-dependent.** E4B: AUC 1.000, 100% detection at 1% FPR on ~390 tokens. 31B: AUC 0.946, only **51%** detection at the same length (§5.1a) |
| It is invisible to a different key | ✅ | Wrong-key AUC 0.45–0.60 (chance); mean score 0.503 either way |
| It does not degrade quality | ✅ | On **both models**: fluency, diversity and GSM8K deltas all statistically indistinguishable from zero, and opposite-signed across the two. Specific to SynthID — see §3, this does *not* transfer to the green-list scheme |
| False positives on human writing are controlled | ✅ | 1.3% observed at a nominal 1% threshold, 0% at 0.1% |
| It works on short text | ❌ | 59% detection at 100 tokens, 18% at 25 |
| It works on code and structured output | ❌ | AUC 0.77 on code, 0.55 on JSON — despite ample length |
| It survives paraphrasing | ❌ | AUC collapses 1.000 → 0.607; detection rate 0% |
| **Latency** is essentially unaffected | ✅ | +1.7 ms per token; 1.7% (31B) to 4.2% (E4B) of a decode step |
| **Batched throughput** is unaffected | ⚠️ | 39–57% loss at batch 16–32 — but this is an artifact of the reference logits processor, not of the method (§5.8) |
| Detection needs no GPU | ✅ | 1,083 texts/s on CPU (0.9 ms/text) — **but only after the fix in §5.9**; the upstream code makes CPU detection of GPU-generated text impossible |

The three ❌ rows are **inherent to the method**: they follow from the fact that a watermark
can only ride on randomness the model already had. No implementation will fix them, and they
define what the control can and cannot be used for.

The ⚠️ row is the opposite — a **fixable engineering problem** in the reference logits
processor, quantified in §5.8. It should not be read as a cost of watermarking.

### 5.1 Detection at full length

| Suite | Median tokens scored | AUC | 95% CI | TPR @1% FPR | Mean score (WM) | Mean score (plain) |
|---|---|---|---|---|---|---|
| creative | 394 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5446 | 0.4999 |
| open_ended | 390 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5256 | 0.5001 |
| financial | 380 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5229 | 0.4988 |
| factual | 6 | 0.5772 | [0.483, 0.677] | 0.028 | 0.5249 | 0.5146 |
| structured | 35 | 0.5472 | [0.441, 0.645] | 0.000 | 0.5020 | 0.4992 |
| code | 352 | 0.7692 | [0.696, 0.834] | 0.010 | 0.5047 | 0.4981 |
| **pooled (high-entropy)** | 390 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5310 | 0.4996 |

On free-form prose the watermark is not merely detectable, it is **unmistakable**: every
watermarked text scores above every unwatermarked one, at ~390 tokens.

The `code` row is the important one. It has **352 scored tokens — as many as `creative` —
yet reaches only AUC 0.77**. This isolates the mechanism cleanly: the weakness of
low-entropy text is not that it is short, it is that there is no choice to encode a signal
in. Where the next token is determined, no watermark can exist. `structured` and `factual`
are weak for both reasons at once: low entropy *and* short outputs (35 and 6 scored tokens).

### 5.1a Detection is weaker on the larger model

The obvious assumption is that a watermark behaves the same way on any model from the same
family. It does not:

| Suite | E4B AUC | E4B TPR@1% | E4B mean g | 31B AUC | 31B TPR@1% | 31B mean g |
|---|---|---|---|---|---|---|
| creative | 1.0000 | 1.000 | 0.5446 | 0.9868 | 0.802 | 0.5153 |
| open_ended | 1.0000 | 1.000 | 0.5256 | 0.9417 | 0.469 | 0.5102 |
| financial | 1.0000 | 1.000 | 0.5229 | 0.9086 | 0.208 | 0.5075 |
| factual | 0.5772 | 0.028 | 0.5249 | 0.4913 | 0.000 | 0.5127 |
| structured | 0.5472 | 0.000 | 0.5020 | 0.5344 | 0.000 | 0.4992 |
| code | 0.7692 | 0.010 | 0.5047 | 0.6265 | 0.000 | 0.5003 |
| **pooled (high-entropy)** | 1.0000 | 1.000 | 0.5310 | 0.9459 | 0.510 | 0.5110 |

At the same ~390 tokens, the 31B is caught **51%** of the time at a 1% false-positive rate
against E4B's **100%**. The per-token signal is roughly halved — a mean g of 0.5110 against
0.5310, over a null of 0.5000.

The cause follows from §3. The watermark can only ride on randomness the sampler was already
going to spend. A larger, better-trained model is *more confident*: its next-token
distributions are more peaked, fewer candidates are near-tied, and there is less residual
entropy for the tournament to steer. Higher capability means lower output entropy means a
weaker watermark. Note this is not a length effect — both corpora are ~390 scored tokens.

The length curve shifts accordingly, and the gap is widest exactly where operational
thresholds get set:

| Target tokens | E4B AUC | E4B TPR@1% | 31B AUC | 31B TPR@1% |
|---|---|---|---|---|
| 25 | 0.9012 | 0.458 | 0.7181 | 0.219 |
| 50 | 0.9497 | 0.625 | 0.8154 | 0.260 |
| 100 | 0.9914 | 0.948 | 0.8910 | 0.219 |
| 200 | 1.0000 | 1.000 | 0.9472 | 0.438 |
| 300 | 1.0000 | 1.000 | 0.9685 | 0.635 |
| 400 | 1.0000 | 1.000 | 0.9868 | 0.802 |

**The uncomfortable implication: watermark strength degrades as models improve.** Detection
thresholds calibrated on one model do not transfer to a stronger one, and a policy tuned
today will silently weaken when the serving model is upgraded. Any deployment should
re-measure detection power per model, and treat the minimum-length rule as model-specific
rather than a property of the scheme.

What does *not* change with model size: key isolation still sits at chance (wrong-key AUC
0.37–0.51), and false positives on human text are identical, because both depend on the key
and the human corpus rather than on the generator.

### 5.2 How much text is needed

| Target tokens | Median scored | AUC | TPR @1% FPR |
|---|---|---|---|
| 25 | 21 | 0.7309 | 0.181 |
| 50 | 46 | 0.8453 | 0.326 |
| 100 | 96 | 0.9582 | 0.594 |
| 200 | 195 | 0.9961 | 0.847 |
| 300 | 293 | 0.9998 | 0.993 |
| 400 | 388 | 1.0000 | 1.000 |

*(averaged over the creative, open_ended and financial suites)*

This is the table that should govern a deployment. At a 1% false-positive rate the detector
finds fewer than 6 in 10 watermarked documents at 100 tokens, and fewer than 2 in 10 at 25.

**Recommended policy, for E4B: issue no verdict below ~100 scored tokens, and treat 200+ as
the range where a negative result carries real weight.** The service enforces a hard floor at
40 tokens and returns `text_too_short` rather than a number.

**These thresholds are model-specific and must be re-derived per model.** On the 31B the same
100-token point yields only 22% detection rather than 95% (§5.1a); no length in the range
tested reaches E4B's performance.

The weighted-mean detector is consistently stronger than the flat mean at exactly the lengths
where it matters — at 25 tokens on `open_ended`, AUC 0.710 vs 0.627 — and identical once text
is long enough for both to saturate.

### 5.3 Key isolation

| Suite | AUC (wrong key) | Mean score (WM text) | Mean score (plain) |
|---|---|---|---|
| creative | 0.5969 | 0.5043 | 0.5027 |
| open_ended | 0.4549 | 0.5036 | 0.5040 |
| financial | 0.5190 | 0.5041 | 0.5038 |
| factual | 0.5155 | 0.5098 | 0.5101 |
| structured | 0.5260 | 0.5056 | 0.5060 |
| code | 0.4903 | 0.5031 | 0.5032 |

A detector holding the wrong key sees nothing: AUC scatters around 0.5, and watermarked and
unwatermarked text score identically (0.503 vs 0.503). Different desks or business units can
hold different keys, and a detection result is scoped strictly to the key that produced it.

### 5.4 False positives on human writing

300 WikiText-103 passages, median 132 scored tokens, mean score 0.4985 (null expectation 0.5).

| Target FPR | Observed (empirical threshold) | Observed (analytic p-value) |
|---|---|---|
| 10.0% | 0.1067 | 0.0833 |
| 1.0% | 0.0333 | 0.0133 |
| 0.1% | 0.0300 | 0.0000 |

Human prose scores 0.4985 on average against a null expectation of 0.5000 — the detector is
unbiased on writing it has never seen.

**An unexpected result: the analytic threshold beats the empirical one.** Calibrating on 300
human texts gives 3.3% false positives at a nominal 1% target, while the closed-form p-value
gives 1.3%. The reason is sample size, not principle: 300 texts split across length buckets
leaves ~150 per bucket, and a 1-in-1000 quantile cannot be estimated from 150 samples.
Empirical calibration is the right approach *given enough data* — on the order of 10⁴ texts
per length bucket — and actively harmful below that.

The analytic null was checked directly against 516 unwatermarked model outputs and is
well-calibrated and slightly conservative:

n = 516

| Nominal p threshold | Observed rate |
|---|---|
| 0.10 | 0.0814 |
| 0.01 | 0.0019 |
| 0.001 | 0.0000 |

**Token round-trip:** 66.8% of texts re-tokenise exactly; 99.5% of token positions agree.

**Use the analytic p-value unless you can calibrate at scale.**

### 5.5 Quality and accuracy

Fluency, scored by **Qwen3.8-27B** — a model from an unrelated family, with its own
architecture, tokenizer and training data, so it shares none of Gemma's idiosyncrasies:

| Model | Suite | n pairs | PPL watermarked | PPL plain | Difference [95% CI] | Significant? |
|---|---|---|---|---|---|---|
| Gemma-4-E4B | creative | 96 | 7.261 | 7.389 | -0.128 [-0.394, +0.134] | no |
| Gemma-4-E4B | open_ended | 96 | 3.424 | 3.369 | +0.055 [-0.015, +0.129] | no |
| Gemma-4-E4B | financial | 96 | 3.302 | 3.303 | -0.001 [-0.102, +0.099] | no |
| Gemma-4-31B | creative | 96 | 5.678 | 5.611 | +0.066 [-0.114, +0.249] | no |
| Gemma-4-31B | open_ended | 96 | 2.808 | 2.844 | -0.036 [-0.089, +0.018] | no |
| Gemma-4-31B | financial | 96 | 2.825 | 2.764 | +0.061 [-0.026, +0.154] | no |

Diversity and degeneracy, paired by prompt:

| Model | Suite | n pairs | distinct-2 WM | distinct-2 plain | Difference [95% CI] |
|---|---|---|---|---|---|
| Gemma-4-E4B | creative | 96 | 0.9628 | 0.9640 | -0.0012 [-0.0046, +0.0022] |
| Gemma-4-E4B | open_ended | 96 | 0.9596 | 0.9602 | -0.0006 [-0.0041, +0.0028] |
| Gemma-4-E4B | financial | 96 | 0.9350 | 0.9408 | -0.0057 [-0.0119, +0.0003] |
| Gemma-4-E4B | factual | 76 | 0.9774 | 0.9801 | -0.0027 [-0.0075, +0.0030] |
| Gemma-4-E4B | structured | 64 | 0.9621 | 0.9622 | -0.0000 [-0.0040, +0.0045] |
| Gemma-4-E4B | code | 96 | 0.9005 | 0.9007 | -0.0002 [-0.0090, +0.0084] |
| Gemma-4-31B | creative | 96 | 0.9539 | 0.9533 | +0.0006 [-0.0024, +0.0036] |
| Gemma-4-31B | open_ended | 96 | 0.9513 | 0.9528 | -0.0015 [-0.0046, +0.0016] |
| Gemma-4-31B | financial | 96 | 0.9437 | 0.9437 | -0.0000 [-0.0049, +0.0046] |
| Gemma-4-31B | factual | 85 | 1.0000 | 0.9989 | +0.0011 [+0.0000, +0.0029] |
| Gemma-4-31B | structured | 64 | 0.9666 | 0.9672 | -0.0006 [-0.0022, +0.0006] |
| Gemma-4-31B | code | 96 | 0.9249 | 0.9272 | -0.0023 [-0.0083, +0.0032] |

Task accuracy on sampled chain-of-thought:

| Model | Watermarked | Unwatermarked | Difference [95% CI] | Flips WM-only / plain-only |
|---|---|---|---|---|
| Gemma-4-E4B | 156/250 (0.624) | 161/250 (0.644) | **-0.0200** [-0.1037, +0.0641] | 15 / 20 |
| Gemma-4-31B | 233/250 (0.932) | 230/250 (0.920) | **+0.0120** [-0.0351, +0.0595] | 8 / 5 |

No measure shows a degradation, on either model. Every confidence interval contains zero.

The GSM8K result is the most informative, because the two models disagree in *direction*:
the watermarked arm is 2.0 pp **worse** on E4B and 1.2 pp **better** on the 31B. A real
systematic cost would push the same way on both. Two opposite-signed, non-significant
deltas, with near-symmetric item-level flips (15/20 and 8/5), are what sampling noise at
temperature 1.0 looks like.

The 31B interval is also tighter (±4.7 pp vs ±8.4 pp) despite the same 250 problems, because
its accuracy sits near ceiling (93% vs 62%) where binomial variance is smaller.

Two honest caveats about strength of evidence:

- The GSM8K intervals are **wide** (±8 pp on E4B, ±4.7 pp on the 31B). They rule out a large
  accuracy loss, not a small one. The fluency and diversity measures are far tighter and
  carry most of the weight.
- Perplexity is a proxy for fluency, not a measure of usefulness. It would not detect a
  failure mode that leaves text fluent but less helpful, and no automatic metric would.
  Human side-by-side rating remains the stronger design.

**A methodological note.** Multiple-choice benchmarks scored by comparing option
log-probabilities — MMLU, HellaSwag, ARC — are **unaffected by watermarking by
construction**, because no sampling occurs and the logits processor is never invoked.
Reporting "no MMLU delta" would be measuring nothing at all. Only sampled, free-form
generation can be affected, which is why GSM8K with sampled chain-of-thought was used.

### 5.6 Robustness

| Attack | Level | Median tokens | AUC | TPR @1% FPR | Mean score (WM) |
|---|---|---|---|---|---|
| none | - | 392 | 1.0000 | 1.000 | 0.5351 |
| lowercase | - | 389 | 0.9941 | 0.891 | 0.5233 |
| strip_formatting | - | 370 | 0.9999 | 0.995 | 0.5307 |
| truncate | 0.75 | 285 | 0.9995 | 0.969 | 0.5316 |
| truncate | 0.5 | 187 | 0.9932 | 0.896 | 0.5298 |
| truncate | 0.25 | 90 | 0.9433 | 0.589 | 0.5270 |
| truncate | 0.125 | 42 | 0.8625 | 0.276 | 0.5248 |
| delete_words | 0.05 | 364 | 0.9974 | 0.979 | 0.5272 |
| delete_words | 0.1 | 345 | 0.9910 | 0.911 | 0.5228 |
| delete_words | 0.2 | 307 | 0.9710 | 0.682 | 0.5153 |
| delete_words | 0.4 | 230 | 0.7575 | 0.036 | 0.5065 |
| swap_words | 0.05 | 384 | 0.9985 | 0.974 | 0.5247 |
| swap_words | 0.1 | 384 | 0.9921 | 0.859 | 0.5184 |
| swap_words | 0.2 | 384 | 0.9054 | 0.271 | 0.5099 |
| swap_words | 0.4 | 385 | 0.7196 | 0.021 | 0.5024 |
| substitute_words | 0.05 | 381 | 0.9989 | 0.953 | 0.5263 |
| substitute_words | 0.1 | 378 | 0.9929 | 0.911 | 0.5212 |
| substitute_words | 0.2 | 371 | 0.9585 | 0.703 | 0.5130 |
| substitute_words | 0.4 | 360 | 0.7491 | 0.188 | 0.5036 |
| mix_with_human | 0.75 | 378 | 0.9983 | 0.969 | 0.5239 |
| mix_with_human | 0.5 | 324 | 0.9783 | 0.688 | 0.5174 |
| mix_with_human | 0.25 | 226 | 0.8715 | 0.328 | 0.5108 |
| mix_with_human | 0.1 | 170 | 0.7034 | 0.042 | 0.5049 |
| paraphrase | 1 | 394 | 0.6074 | 0.000 | 0.5010 |
| paraphrase | 2 | 391 | 0.6451 | 0.016 | 0.5012 |
| round_trip_translate | fr | 288 | 0.9788 | 0.547 | 0.5163 |

*(unattacked baseline AUC = 1.0000)*

Every attack was applied to **both** arms and the AUC recomputed on the attacked pair, so
these numbers isolate damage to the watermark from changes to the score distribution.

Reading the table:

- **Free edits are free.** Lowercasing and stripping markdown cost almost nothing (AUC 0.994
  and 0.9999), even though they re-tokenise much of the text.
- **Casual editing survives.** Deleting, swapping or substituting up to 10% of words leaves
  AUC above 0.99. It degrades at 20% and collapses at 40% — but by then the text is visibly
  damaged, so a real editor would not go there.
- **Dilution behaves sensibly.** A document that is 25% model-written scores AUC 0.87 and is
  flagged 33% of the time. A weak signal on a mostly-human document is the *correct* answer,
  not a failure.
- **Paraphrase defeats it completely.** One pass through another model takes AUC from 1.000
  to 0.607 and detection at a 1% false-positive rate from 100% to **0%**. A second pass adds
  nothing because the first already reached the floor.
- **But round-trip translation does not.** EN→FR→EN retains AUC 0.979 and 55% detection.

That last contrast is the most useful finding here. "Passed through another model" is not
what destroys the watermark — *deliberate rewording* is. Back-translation reconstructs
similar phrasing and therefore similar token sequences; a paraphrase prompt explicitly asks
for different ones. The practical implication: the watermark survives ordinary round-tripping
through tools, and fails against anyone actively trying to remove it.

### 5.7 Inference cost: latency vs. throughput

These are two different quantities with two different answers, and conflating them is easy.

**Latency — what a single user experiences — is essentially unaffected.** At batch 1 the
watermark adds about 1.7 ms per token: 4.2% of a decode step on E4B, 1.7% on the 31B. Nobody
notices this.

**Batched throughput is a different story**, and the raw numbers look alarming:

*Throughput in tokens per second — higher is faster.*

| Batch size | Gemma-4-E4B plain (tok/s) | Gemma-4-E4B watermarked (tok/s) | Gemma-4-E4B overhead | Gemma-4-31B plain (tok/s) | Gemma-4-31B watermarked (tok/s) | Gemma-4-31B overhead |
|---|---|---|---|---|---|---|
| 1 | 26.9 | 25.8 | **4.2%** | 16.0 | 15.7 | **1.7%** |
| 4 | 99.9 | 87.2 | **12.7%** | 61.2 | 56.5 | **7.7%** |
| 16 | 398.0 | 241.4 | **39.4%** | 210.0 | 156.3 | **25.6%** |
| 32 | 770.4 | 329.0 | **57.3%** | 341.5 | 213.7 | **37.4%** |

Read the plain and watermarked columns as speeds: the plain figure is higher because it is
*faster*. Watermarking adds work at every decoding step, so fewer tokens come out per second,
and the overhead column is that shortfall as a percentage. At batch 32 on E4B, 770 tok/s
becomes 329 tok/s — the same GPU produces well under half as much text per second.

Overhead climbs steeply with batch size on both models, and is consistently ~35% lower on
the 31B than on E4B. That much is a real measurement. But it does *not* mean the method is
expensive, and the next table shows why.

#### Where the cost actually goes

Timing the SynthID logits processor in isolation, with no model involved, accounts for
essentially all of it:

| Batch | Model forward (ms) | Watermarked step (ms) | Observed Δ | Processor alone | Explained |
|---|---|---|---|---|---|
| 1 | 37.18 | 38.82 | 1.63 | 1.76 | **108%** |
| 4 | 40.05 | 45.86 | 5.81 | 5.43 | **94%** |
| 16 | 40.20 | 66.29 | 26.09 | 26.14 | **100%** |
| 32 | 41.54 | 97.26 | 55.72 | 55.88 | **100%** |

The overhead is **100% the logits processor**. And its scaling is the whole story:

| Batch | Processor ms/step | Processor ms **per sequence** |
|---|---|---|
| 1 | 1.76 | 1.76 |
| 4 | 5.43 | 1.36 |
| 16 | 26.14 | 1.63 |
| 32 | 55.88 | 1.75 |

**The model forward is nearly flat across batch sizes (37 → 42 ms) because GPUs batch
efficiently. The watermark processor is perfectly linear — a fixed ~1.7 ms per sequence that
never amortises.** That is not a property of SynthID. It is a property of this
implementation: `update_scores` runs a Python `for i in range(depth)` loop, plus `vmap`
calls, serialised per sequence over a 262k-wide tensor.

The depth sweep confirms the loop is the cost:

| Depth | Processor ms/step |
|---|---|
| 1 | 0.54 |
| 5 | 3.11 |
| 10 | 7.18 |
| 30 | 26.16 |

Roughly 0.87 ms per depth level at batch 16 — linear in the number of Python iterations.

#### What this implies

If the processor amortised across a batch the way the model forward does, the overhead would
be flat at about 4% everywhere instead of climbing to 57%:

| Batch | Measured overhead | If it batched like the model |
|---|---|---|
| 1 | 4.2% | **4.5%** |
| 4 | 12.7% | **4.2%** |
| 16 | 39.4% | **4.2%** |
| 32 | 57.3% | **4.1%** |

So the honest conclusion is:

- **Latency is genuinely minimal**, and the widely-repeated claim to that effect is correct.
- **The batched-throughput cost measured here is real but avoidable.** It is the price of an
  unoptimised reference implementation, not of watermarking. A fused kernel that computes all
  depths in one pass, or a serving stack (vLLM, TGI) with an optimised processor, should
  recover most of the ~4%-vs-57% gap.
- **Until you have that, depth is the lever.** Depth 10 costs 15.2% instead of 39.4% on E4B
  (8.6% vs 25.6% on the 31B), and detection already saturates at AUC 1.000 by 200 tokens with
  depth 30. Depth is part of the key, so it must be fixed before keys are issued.

Measured end to end on both models, the depth lever looks like this:

*Watermarked throughput at each depth, tokens per second — higher is faster.*

| Depth | Gemma-4-E4B (tok/s) | Gemma-4-E4B overhead | Gemma-4-31B (tok/s) | Gemma-4-31B overhead |
|---|---|---|---|---|
| 1 | 392.7 | **1.3%** | 208.8 | **0.6%** |
| 5 | 369.1 | **7.3%** | 201.7 | **4.0%** |
| 10 | 337.7 | **15.2%** | 192.0 | **8.6%** |
| 30 | 241.1 | **39.4%** | 156.3 | **25.6%** |

**Before deploying, benchmark your own serving stack.** These numbers characterise
HuggingFace `generate()`; they are not a property of SynthID, and they should not be used to
argue either for or against watermarking on a different stack.

#### Detection costs almost nothing

| Model | Device | Texts/s | ms per text |
|---|---|---|---|
| Gemma-4-E4B | cpu | 1082.6 | 0.9 |
| Gemma-4-E4B | cuda:0 | 1241.9 | 0.8 |
| Gemma-4-31B | cpu | 1255.8 | 0.8 |
| Gemma-4-31B | cuda:0 | 1257.2 | 0.8 |

Detection is a hash and a mean. It needs the tokenizer, not the model, and CPU is within 13%
of GPU on E4B and indistinguishable on the 31B — so the detection service needs no
accelerator at all.

### 5.8 A bug worth knowing about

Upstream Transformers builds the watermark's g-value sampling table with a **device-local
RNG**:

```python
generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
self.sampling_table = torch.randint(0, 2, (size,), generator=generator, device=device)
```

`torch.randint` draws different values from CUDA and CPU generators given the same seed. The
consequence is that **the same key produces a different watermark depending on which device
the processor was built on.** Text generated on a GPU is invisible to a detector running on
CPU — and the failure is silent: the detector returns an ordinary-looking null score, not an
error.

This was found the first time generation was run on GPU and detection on CPU: the
watermarked text scored 0.4974 (p = 0.69), indistinguishable from unwatermarked.

Since the sampling table is the *only* source of randomness in the algorithm — the rest is a
deterministic linear-congruential hash over int64 — the fix is to draw the table on CPU and
move it to the target device. `synthmark` does this for both generation and detection, so
they cannot disagree. Two regression tests cover it: one asserting g-values, masks and scores
match across devices, and one documenting the upstream behaviour so we learn when it is fixed
upstream.

Anyone deploying SynthID via Transformers should check this. It would not show up in a
single-machine test, and it silently breaks CPU detection services and mixed-hardware fleets.

### 5.9 Worked example

Generated on GPU with key `markets-research/v1`, detected on **CPU** — which only works
because of the fix above.

> When interest rates in the broader economy rise, bond prices typically move in the opposite
> direction, which can seem counterintuitive at first. The core reason lies in the
> relationship between the fixed coupon payments a bond provides and the new, higher
> prevailing interest rates. […]

| Case | Score | Tokens | z | p |
|---|---|---|---|---|
| watermarked, correct key | 0.5218 | 310 | +4.21 | 1.3 × 10⁻⁵ |
| watermarked, **wrong** key | 0.4906 | 310 | −1.80 | 0.96 |
| unwatermarked, correct key | 0.5031 | 316 | +0.60 | 0.28 |

The watermarked and unwatermarked outputs read identically well; the difference lives
entirely in the statistics of which near-equivalent words were chosen.

### 5.10 Token round-trip

The watermark lives in *token* choices, but a detector is handed *text* and must re-tokenise
it. Of 400 watermarked texts, only **66.8% re-tokenise to exactly the original ids** — but
**99.5% of individual token positions agree**. The disagreements are isolated boundary
effects, and the cost to detection is negligible. Worth knowing, not worth engineering
around.

---

## 6. The package

`synthmark` is the reusable half of this work. It wraps the Transformers SynthID
implementation and adds what a deployment needs.

| Module | What it provides |
|---|---|
| `keys.py` | HKDF-SHA256 key derivation from one master secret, fingerprints, 0600 storage |
| `config.py` | Bridge to the HF API, plus the device-portability fix |
| `generate.py` | Watermarked / unwatermarked generation, perplexity scoring |
| `detect.py` | g-values, masking, three scoring methods, empirical calibration |
| `bayesian.py` | Training and use of the learned detector (optional; not used by default) |
| `attacks.py` | Truncation, editing, paraphrase, translation, dilution |
| `metrics.py` | AUC, TPR@FPR, bootstrap and Newcombe confidence intervals |
| `serve.py` | FastAPI detection service, multi-key |
| `cli.py` | `keygen` · `generate` · `detect` · `calibrate` · `serve` |

Turning the watermark on is a one-argument change at the call site:

```python
out = lm.generate(prompts, key=key)      # watermarked
out = lm.generate(prompts, key=None)     # ordinary HF path
```

Three design decisions worth calling out, because they are the ones that keep the
system honest rather than merely working:

**Greedy decoding is refused, not silently unmarked.** With no sampling there is no
randomness to encode a signal in. Passing `do_sample=False` together with a key raises
rather than returning confidently unmarked text.

**The service refuses a verdict below ~40 scored tokens.** Returning a
confident-looking number on two sentences is how false accusations happen.

**Every negative result is caveated in the response body.** "No watermark detected" is
not "a human wrote this", and the API says so in the payload rather than in
documentation someone will not read.

### Key management

Keys are derived, not distributed:

```python
key = derive_key(master_secret, "markets-research/v1")
```

One escrowed master secret yields unlimited independent keys. Rotation is a new label.
Any authorised host can reconstruct any key, so key material never travels.

The critical point for a security review: **the key is signing-grade material**. Whoever
holds it can detect the watermark *and forge it*. The obvious way to let a downstream team
check text — hand them the key — also hands them the ability to fabricate text your own
detector will confirm as yours. Run detection as a service instead. See
[`docs/key-management.md`](docs/key-management.md).

---

## 7. Limitations

Stated plainly, because a detector that is oversold is worse than no detector.

- **It is not proof of authorship.** It answers "is this text statistically consistent
  with this key", nothing more. It cannot distinguish text the model wrote from text the
  model edited or a human lightly revised.
- **A negative is not evidence of human authorship.** Short, edited, paraphrased, or
  low-entropy text all produce negatives, as does any text from a different model.
- **Paraphrase defeats it.** Anyone who wants to remove the watermark can, cheaply.
  It is a provenance signal for ordinary use, not an anti-abuse control against a
  motivated adversary.
- **It requires the matching tokenizer.** Detection is defined over token sequences.
- **It is tied to the sampling configuration.** Greedy or very low-temperature decoding
  embeds little or nothing.
- **Findings here are for one model.** Detection strength depends on the model's output
  entropy; a different model needs its own measurement.

---

## 8. Mapping the findings to the obligations

§1 states the legal requirement. This section maps what was measured onto it, obligation by
obligation, so a compliance reader can see which claims the evidence supports.

| Art. 50(2) requirement | Status | Evidence |
|---|---|---|
| Output "marked in a machine-readable format" | **Met** | The mark is embedded at generation; no side-channel or metadata needed |
| "Detectable as artificially generated" | **Met, with conditions** | AUC 1.000 / 100% detection on E4B long-form prose; **0.946 / 51% on the 31B** (§5.1a) |
| "Effective… as far as technically feasible" | **Bounded, and quantified** | Limits measured and stated: length (§5.2), entropy (§5.1), paraphrase (§5.6) |
| "Robust and reliable" | **Partially** | Survives editing, formatting, translation; **not** paraphrase (§5.6) |
| Marking carries no personal data | **Met by design** | The key is per-deployment, not per-user or per-session (§5.3) |
| Marking does not degrade the service | **Met** | No quality cost on either model (§5.5); latency +1.7 ms/token (§5.8) |

**The three numbers a governance owner should actually sign off on** — not AUC, which
averages over operating points nobody would deploy:

1. **False-positive rate at the deployed threshold**, measured on human writing — 1.3% at a
   nominal 1% target (§5.4). This is the rate at which a person's own writing gets flagged.
2. **Minimum text length below which no verdict is issued** — the service refuses under 40
   scored tokens; §5.2 argues for ~100 as a policy floor on E4B.
3. **Per-model detection power**, because it does not transfer between models (§5.1a) and
   will weaken silently when the serving model is upgraded.

**Explicitly outside what this control can support:** detection of paraphrased content,
attribution to a person, and any claim about content the pipeline did not generate. For
images and files the appropriate mechanism is C2PA content credentials — a different,
complementary control.

A closing caution for anyone drafting the compliance narrative. Article 50(4)'s exemption
for human-edited text with a named editorial owner is a *process* control, and for many
internal workflows it is both cheaper and more defensible than a technical mark. Watermarking
is the right answer for machine-readable provenance at scale; it is not automatically the
right answer to every transparency obligation.

---

## 9. Reproducing

```bash
pip install -e '.[serve,dev]'
pytest                                   # unit tests, no model download

cd experiments
./run_all.sh                             # full evaluation
```

Individual studies:

```bash
python 01_generate_corpus.py --samples-per-prompt 4
python 02_detectability.py
python 03_quality.py --gsm8k-n 250
python 04_robustness.py
python 05_overhead.py
python 06_bayesian.py
```

Raw results are written to `results/*.json`; the logs alongside them contain the formatted
tables reproduced in this report.
