# Detailed reproducibility guide

## Experimental objects

- `M`: the XGBoost black-box classifier.
- `T0`: the selected decision-tree surrogate.
- `R0`: the ordered rules extracted from `T0`.
- `Faithful-Refined`: verifier-accepted edits proposed by Qwen from
  black-box/rule counterexamples on `D_refine`.
- `Corrupted-R0`: a negative control made by shuffling `R0` consequents.
- `P`: Qwen acting as a prediction proxy under each explanation condition.

## Split contract

Stage 01 creates one immutable, stratified assignment with seed `2027`:

| Split | Fraction | Permitted use |
|---|---:|---|
| `train_bb` | 0.60 | fit preprocessing state and `M` |
| `val_bb` | 0.10 | black-box validation reporting |
| `sur` | 0.10 | fit `T0`; fit LIME/Anchors background state |
| `refine` | 0.10 | choose surrogate depth; propose and verify edits |
| `eval` | 0.10 | final rule, proxy, and counterfactual evaluation |

Every later stage joins data to the saved assignment by `row_id`. Changing the
seed or fractions defines a different experiment.

## Preprocessing contract

All 23 HELOC predictors are numeric. Values `-9`, `-8`, and `-7` are treated as
special/missing. For every feature, Stage 01:

1. Computes a median on `train_bb` only.
2. Replaces special/missing values with that median.
3. Adds a `<feature>__is_special` indicator.
4. Saves the state and feature schema for reuse by every later stage.

No later stage refits preprocessing.

## Stage reference

### Stage 00 — environment check

```bash
python scripts/00_check_environment.py
```

Checks all core imports and reports versions. LIME and Alibi are reported as
optional because the primary rule pipeline can run without stages 09 and 10.

### Stage 01 — split and preprocess

```bash
python scripts/01_make_splits.py --config configs/heloc_cgrr.yaml
```

Writes:

- `splits/split_assignments_seed2027.csv`
- `splits/split_counts.csv`
- `splits/split_class_counts.csv`
- `preprocess/preprocessor_state.json`
- `preprocess/feature_schema.json`
- `preprocess/preprocess_summary.json`

### Stage 02 — black-box model

```bash
python scripts/02_train_blackbox.py --config configs/heloc_cgrr.yaml
```

Fits XGBoost on `train_bb`, then writes the serialized model, predictions, model
configuration, and per-split metric files. These are generated artifacts and
must remain untracked.

### Stage 03 — surrogate and initial rules

```bash
python scripts/03_build_surrogate.py --config configs/heloc_cgrr.yaml
python scripts/03b_make_rule_baselines.py --config configs/heloc_cgrr.yaml
```

Each candidate depth is fitted on `sur` and scored on `refine` with the
configured fidelity/coverage/complexity objective. The selected tree is exported
as ordered `R0` JSON and readable text. The second command creates the
consequent-shuffled negative control.

### Stage 04 — verifier-constrained refinement

```bash
python scripts/04_refine_rules.py \
  --config configs/heloc_cgrr.yaml \
  --source faithfulness \
  --output-name Faithful-Refined
```

For each weak rule, Qwen receives the current rules, feature schema, statistics,
and bounded `refine` examples. JSON candidates are parsed, structurally checked,
executed, scored, and accepted only when all configured constraints and the
minimum objective improvement hold. Prompts and candidate decisions are saved
under the configured output directory for private audit.

Proxy-guided refinement first creates `R0` forward predictions on `refine`:

```bash
python scripts/06_proxy_simulation.py \
  --config configs/heloc_cgrr.yaml \
  --condition R0 \
  --mode forward \
  --split refine

python scripts/04_refine_rules.py \
  --config configs/heloc_cgrr.yaml \
  --source proxy \
  --output-name Proxy-Refined
```

These predictions are stored under `refinement_inputs/`, separately from the
`proxy/` directory read by final evaluation.

### Stage 05 — counterfactual pairs

```bash
python scripts/05_make_counterfactuals.py --config configs/heloc_cgrr.yaml
```

For each held-out row, the script searches transformed `eval` rows for nearest
real same-label and opposite-label neighbors under the black-box labels. It does
not synthesize credit records.

### Stage 06 — proxy simulation

```bash
python scripts/06_proxy_simulation.py \
  --config configs/heloc_cgrr.yaml \
  --condition x-only \
  --mode forward
```

Valid rule conditions are `R0`, `Faithful-Refined`, `Corrupted-R0`, and any
additional rule JSON created in `outputs/.../rules/`. Modes are `forward` and
`counterfactual`. JSONL is appended after each item to support resumption; CSV
is a cleaned convenience export.

### Stage 07 — evaluation

```bash
python scripts/07_evaluate.py --config configs/heloc_cgrr.yaml
```

Computes rule fidelity, coverage, conflict, per-rule diagnostics, symbolic
executor predictions, proxy summaries, and paired bootstrap contrasts. It reads
all available rule JSON and final proxy CSV files from the configured output
directory.

### Stage 08 — table sources

```bash
python scripts/08_make_tables.py --config configs/heloc_cgrr.yaml
```

Converts metric artifacts into rounded CSV and LaTeX source. No generated table
belongs in the public code-only repository.

### Stage 09 — LIME and Anchors extraction

```bash
python scripts/09_extract_lime_anchor_explanations.py \
  --config configs/heloc_cgrr.yaml \
  --explainer both \
  --mode both
```

Fits explainer state on `sur`, explains `eval` items, and stores structured and
text forms. Current-format records carry a version field; legacy records that
might disclose a row-specific target are ignored.

### Stage 10 — local-explanation proxy conditions

```bash
python scripts/10_proxy_local_explanations.py \
  --config configs/heloc_cgrr.yaml \
  --condition LIME \
  --mode forward
```

Conditions are `LIME` and `Anchors`; modes are `forward` and `counterfactual`.
The loader rejects missing, legacy, empty, or target-disclosing explanation
text.

## Output layout

All paths below are relative to `project.output_dir`:

| Directory | Generated content |
|---|---|
| `counterexamples/` | rule/counterexample diagnostic rows |
| `evaluation_items/` | counterfactual pair definitions |
| `local_explanations/` | LIME/Anchors structured explanations |
| `logs/` | refinement records and stage logs |
| `metrics/` | raw metric and bootstrap CSV files |
| `models/` | fitted black-box model and parameters |
| `predictions/` | black-box predictions |
| `preprocess/` | immutable preprocessing state and schema |
| `prompts/` | refiner prompts used in a run |
| `proxy/` | evaluation-split proxy responses |
| `refinement_inputs/` | non-evaluation proxy responses |
| `rules/` | extracted and refined rule sets |
| `splits/` | row-level split assignment |
| `tables/` | generated CSV and LaTeX table sources |

## Failure and restart behavior

- Each launcher exits on the first failing command.
- A stage marker is written only after its command succeeds.
- Proxy and explanation stages append one JSONL record per item and skip keys
  already present, allowing item-level resumption.
- `FORCE=1` bypasses stage markers. It does not delete existing JSONL. Use a new
  output directory for a completely independent rerun.
- Never merge partial smoke-run artifacts with a full run.

## What to record privately for a paper run

Record the repository commit, dataset hash, full YAML file, package versions,
Python version, GPU model, CUDA version, and start/end timestamps. Keep hostnames,
usernames, filesystem roots, scheduler accounts, queue/partition names, job IDs,
and raw scheduler logs out of the public artifact.

