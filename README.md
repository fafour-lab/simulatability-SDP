# HELOC counterexample-guided rule refinement

This repository is the code-only reproducibility artifact for the HELOC
experiments. It recreates the data split, black-box model, initial rules,
counterexample-guided refinement, proxy-simulatability conditions, local
explanation baselines, evaluation files, and paper-table source files.

The repository intentionally contains no reported results, generated tables,
run logs, paper sources, scheduler configuration, machine-specific paths,
container images, model checkpoints, or downloaded LLM weights. All generated
artifacts are written under `outputs/` or `logs/`, both of which are ignored by
Git.

## Repository contents

```text
.
├── cgrr/                         Python implementation
│   ├── llm/                      Qwen prompts, parsing, and inference
│   ├── metrics/                  summaries and paired bootstrap estimates
│   ├── rules/                    rule DSL, execution, extraction, verification
│   └── utils/                    file and logging helpers
├── configs/heloc_cgrr.yaml       complete experiment configuration
├── data/
│   ├── README.md                 provenance and checksum
│   └── heloc_dataset_v1.csv      anonymized HELOC input
├── docs/                         detailed reproducibility documentation
├── scripts/                      numbered pipeline stages and launchers
├── tests/                        leakage and refinement-input checks
├── pyproject.toml
└── requirements.txt              reference environment versions
```

## Requirements

- Python 3.12
- A CUDA-capable GPU with enough memory for Qwen3-8B for stages 04, 06, and 10
- CPU execution is sufficient for splitting, XGBoost, rule extraction,
  counterfactual construction, local explanation extraction, evaluation, and
  table generation
- Internet access for the first Hugging Face model download, or an already
  available local Qwen3-8B checkpoint

The LLM is loaded with local `transformers` inference. No remote inference API
is used and no credential is required for a public checkpoint.

## Installation

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/00_check_environment.py
```

PyTorch wheels are platform-specific. If the pinned default wheel is not
appropriate for the local CUDA driver, install the corresponding official
PyTorch build first, then install the remaining requirements.

Run the unit tests before an experiment:

```bash
python -m unittest discover -s tests -v
```

## Verify the input

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path

path = Path("data/heloc_dataset_v1.csv")
print(sha256(path.read_bytes()).hexdigest())
PY
```

The expected SHA-256 digest is:

```text
abdb86f415228b6754dcdee791feaf25bb73a9147f1742cf09c9a2dd54308a93
```

See `data/README.md` for provenance and the special-value policy.

## LLM checkpoint configuration

The default configuration names `Qwen/Qwen3-8B`. Transformers uses the normal
Hugging Face cache and downloads the checkpoint only if it is absent. Downloaded
weights remain outside version control.

For offline use, make a local copy of the YAML configuration, point it at an
existing checkpoint, and disable downloads:

```yaml
qwen:
  model_id: $QWEN_MODEL_PATH
  cache_dir:
  local_files_only: true
  max_new_tokens: 1200
  temperature: 0.0
  top_p: 1.0
  dtype: bfloat16
  device_map: auto
  require_cuda: true
```

Then run:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3-8B
bash scripts/run_pipeline.sh configs/heloc_cgrr_offline.yaml
```

Do not place the checkpoint inside the repository. Common weight and checkpoint
formats are ignored by `.gitignore` as an additional safeguard.

## Run the primary pipeline

```bash
bash scripts/run_pipeline.sh configs/heloc_cgrr.yaml
```

The launcher runs the following stages in order:

1. Create immutable stratified splits and training-only preprocessing state.
2. Train the XGBoost black-box model.
3. Select a decision-tree surrogate and extract `R0`.
4. Create the corrupted-rule negative control.
5. Refine `R0` with verifier-constrained Qwen proposals.
6. Construct nearest-real counterfactual pairs.
7. Run forward and counterfactual proxy-simulatability conditions.
8. Evaluate rule and proxy artifacts.
9. Generate CSV and LaTeX table source files.

The optional proxy-guided refinement condition is enabled with:

```bash
bash scripts/run_pipeline.sh configs/heloc_cgrr.yaml --with-proxy-refinement
```

Each completed stage receives a marker under `logs/pipeline_markers/`. A rerun
skips marked stages. To recompute every stage without manually removing markers:

```bash
FORCE=1 bash scripts/run_pipeline.sh configs/heloc_cgrr.yaml
```

Set `PYTHON_BIN` to select a particular interpreter and `MARKER_DIR` to keep
markers elsewhere. The launchers contain no scheduler- or institution-specific
settings.

## Run LIME and Anchors conditions

After installing all requirements:

```bash
bash scripts/run_local_explanations.sh configs/heloc_cgrr.yaml
```

This launcher creates label-safe LIME and Anchors explanations on evaluation
items, sends those explanations through the same Qwen proxy, and regenerates
the evaluation artifacts. The explainer background distribution is fitted only
on `D_sur`; evaluation rows are never used to fit the LIME discretizer or the
Anchors explainer.

## Small smoke run

The LLM stages accept `--limit` for a cheap integration check after stages
01--05 have completed:

```bash
python scripts/06_proxy_simulation.py \
  --config configs/heloc_cgrr.yaml \
  --condition R0 \
  --mode forward \
  --limit 2

python scripts/09_extract_lime_anchor_explanations.py \
  --config configs/heloc_cgrr.yaml \
  --explainer lime \
  --mode forward \
  --limit 2
```

Use a separate `project.output_dir` in a copied configuration for smoke tests so
partial artifacts cannot be mistaken for a full run.

## Leakage controls

- `D_train_bb` alone fits the preprocessing medians and XGBoost model.
- `D_sur` fits the initial surrogate and the local-explainer background.
- `D_refine` selects the surrogate depth, supplies refinement examples, and
  verifies candidate edits.
- `D_eval` is reserved for final rule and proxy evaluation.
- The black-box uses ground-truth labels; the surrogate and refinement stages
  use black-box labels.
- Proxy-guided refinement reads only forward predictions explicitly marked as
  `data_split=refine`; the loader rejects evaluation-split inputs.
- LIME prompts use one fixed reference class rather than a row-specific target.
- Anchors prompt text withholds the black-box outcome label.

The tests exercise the last three invariants.

## Reproducibility notes

The split seed, model seed, bootstrap seed, preprocessing policy, generation
settings, and experimental hyperparameters are recorded in
`configs/heloc_cgrr.yaml`. LLM decoding is greedy (`temperature: 0.0`), but exact
bitwise equality can still depend on GPU architecture, CUDA libraries, and
low-level kernels. Record the output of `scripts/00_check_environment.py` with
private run records, but do not publish machine paths or scheduler logs.

For the exact stage contracts and artifact names, see
`docs/REPRODUCIBILITY.md`. For the pre-publication exclusion audit, see
`docs/RELEASE_CHECKLIST.md`.

## Dataset and responsible use

HELOC is a financial-risk dataset. The included code is intended to reproduce
the reported research workflow, not to make real lending decisions. Users are
responsible for checking the original dataset terms and applicable legal,
fairness, privacy, and model-risk requirements.

The historical dataset source is the [FICO Explainable Machine Learning
Challenge](https://community.fico.com/s/explainable-machine-learning-challenge).

## License

No software license is asserted by this artifact. Add the authors' chosen
license before publishing the repository. The HELOC data remains subject to its
source terms.

