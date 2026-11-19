# Does a sparse weight edit improve Swedish character predictions?

**Candidate assignment | SE.LLMA | 19 November 2026**

## Research question and hypothesis

Can an interpretable, three-parameter output-bias edit improve Swedish-specific character predictions in a tiny bilingual character language model without large regressions? We hypothesized that increasing the output biases for `å`, `ä`, and `ö` would improve those predictions, but would be indistinguishable from a fixed inference-time logit bias. This is a deliberately narrow mechanistic test, **not** an experiment on a pretrained Swedish LLM.

## Critical reading of Weight-Space Programming

Takehi et al. report successful direct reasoning over small networks (including recognizing the training domain of small character models with input/output layers hidden) and a substantial Qwen3.5-0.8B benchmark improvement after editing 0.016% of weights. Their strongest practical evidence is a model-specific intervention that suppresses reasoning loops, supported by the reported decrease in token-limit exhaustion. This does **not** establish that the model gained new mathematical knowledge. Crucially, some experiments use zero behavioral feedback, whereas the Qwen and Go studies allow up to 50 forward-pass rounds: successful behavior-guided weight optimization is not the same as reading and editing weights from inspection alone. Direct comparisons to prompting, steering, and conventional fine-tuning are limited. Experiment 7 (sycophancy) also gives a useful failure case for raw-weight inspection.

SE.LLMA aims to adapt open language models for accurate, culturally grounded Swedish and build Swedish-specific evaluations. A low-cost localized correction would be relevant, but only if it outperforms simpler controls and preserves other abilities.

## Experiment and controls

A 24-dimensional character embedding, 48-unit single-layer GRU, and linear output were trained for five epochs from scratch on **synthetic, author-written** Swedish and English sentence templates (800 Swedish and 3,200 English training sentences). Independent random combinations generated 240 sentences per language for validation and 400 per language for locked test evaluation. Templates and lexical items *overlap across splits*; our test only measures combinations of known elements. Seed: `20261119`; PyTorch `2.10.0+cpu`.

The edit adds `+1.6` to the three output bias parameters for `å`, `ä`, `ö`. The magnitude was chosen **on validation only** from a 0.0–2.5 grid, minimizing Swedish accented-target next-character NLL while limiting validation English overall NLL degradation to 0.03 nats and Swedish overall NLL degradation to 0.02. Controls: unchanged model, 20 random three-character output-bias edits of the same amplitude, and a fixed inference-time logit offset. Test remained untouched during selection.

## Held-out results (lower NLL is better)

| Metric | Baseline | Targeted edit | Interpretation |
|---|---:|---:|---|
| Swedish accented-target NLL | 0.580 | 0.313 | Better at å/ä/ö targets |
| Swedish accented-target top-1 accuracy | 83.2% | 91.2% | +8.0 percentage points |
| Swedish all-character NLL | 0.488 | 0.509 | **Regression** |
| English all-character NLL | 0.316 | 0.324 | Small regression |

The 20 random edits had mean Swedish NLL `0.534` and English NLL `0.342`; these random controls have different target characters and are not a matched comparator for accent-specific improvement. The edit changed model outputs: for the held-out prefix `^barnet ser ett `, expected `ä`, baseline predicted `g`, and edited predicted `ä`. But it also introduced errors: for `^läraren vänder ett `, expected `g`, baseline predicted `g`, and edited predicted `ä`. More examples are in `examples.json`.

**Key negative result:** the edited model and a constant inference-time logit adjustment yield the **same test metrics**, and their raw logits differ by at most `4.8e-07` in the numerical equivalence check. This is expected algebraically: editing the final bias `b` to `b + δ` gives `Wh + b + δ`, identical to adding `δ` to logits at inference. No special ability to interpret internal representations has been established.

## Interpretation, recommendation and limitations

**Recommendation:** Do **not** prioritize this final-layer bias edit as a new weight-space technique for SE.LLMA. It is less flexible than inference steering, gives no advantage over the exactly equivalent alternative, and worsens overall Swedish NLL even while helping accented targets. The positive result is specific to a known output vocabulary. The experiment does not evaluate grammar, cultural understanding, open-ended generation, real Swedish corpora, or transfer to pretrained transformers. Only one model seed was trained, and examples are synthetic and template-overlapping. There is no statistical uncertainty estimate across model seeds.

**Next experiment:** On a small open-weight Swedish-capable transformer, pre-register a concrete failure mode (e.g., unwanted Swedish-to-English switching), use naturally occurring held-out Swedish data, compare a hidden-layer edit with activation steering and matched prompting, and check Swedish task quality plus English regression across multiple seeds. This would distinguish nontrivial internal weight-space intervention from simple output calibration.

## References and AI usage

1. Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026). *Sparks of Weight-Space Programming*. https://rikiyat.github.io/blog/weight-space-programming/
2. WASP. *SE.LLMA – Swedish large language models*. https://wasp-sweden.org/research/projects/sellma/

**AI usage:** I used ChatGPT to help write and run the experiment code and draft parts of the research note. It also provided input on the experiment design and interpretation of results. All reported measurements come from the included code.