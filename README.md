# SE.LLMA weight-edit pilot

A small and explicitly limited investigation of a sparse output-bias edit for Swedish-specific characters, with held-out data and controls.

## Reproduce

Python 3.10+ with `torch` and `numpy` installed:

```bash
pip install -r requirements.txt
python experiment.py
python build_report.py
```

This rebuilds the synthetic datasets, retrains the small character GRU (seed 20261119), evaluates the baseline/edit/logit-bias/random controls, and regenerates `results.json`, `examples.json`, and `checkpoint.pt`. The experiment **reruns training rather than requiring a pretrained checkpoint**. It writes a local `checkpoint.pt`, which is not committed to this repository. CPU only. Set `torch.set_num_threads(2)` inside the script.

Read `research_note.md` and `research_note.pdf` for interpretation, caveats and references. Outputs are deterministic on the recorded environment to typical numerical tolerances; PyTorch versions/platforms may vary.

## Contents

- `experiment.py`: complete training, edit selection, evaluation, random controls and example extraction
- `results.json`: numerical metrics, configs, seed, random controls
- `examples.json`: held-out corrected predictions and introduced errors
- `checkpoint.pt`: regenerated locally when running `experiment.py` (not committed)
- `research_note.md`, `research_note.pdf`: research note
- `build_report.py`: render a fresh PDF from `research_note.md`; report formatting may differ slightly from the committed PDF
- `requirements.txt`: minimal Python dependencies

## Scientific integrity

The datasets are **synthetic sentence templates**, not natural Swedish. All evaluation uses teacher-forced character prediction. Train/validation/test sentences are generated with different random seeds but use the same lexical inventories and templates. The targeted bias edit is exactly equivalent to a constant inference logit bias. This study does not reproduce the cited paper's headline experiments.

## AI usage

I used ChatGPT to help refine the experiment design, write and run the code, interpret the results, and draft the research note. All reported measurements come from the experiment code included in this repository.

## Scope and submission

The paper and project links are in the research note. This is a limited, CPU-friendly synthetic pilot, **not** a reproduction of the research paper or an intervention on a pretrained Swedish transformer. Reported numbers are descriptive for one model seed, and must not be interpreted as production language-model gains.
