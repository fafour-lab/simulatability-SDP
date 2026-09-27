# Public release checklist

Run this checklist from the repository root before creating a public archive.

## Required contents

- [ ] `cgrr/`, `scripts/`, `tests/`, `configs/`, and documentation are present.
- [ ] `requirements.txt` and `pyproject.toml` are present.
- [ ] The dataset is present only if its redistribution terms permit it.
- [ ] The dataset SHA-256 matches `data/README.md`.
- [ ] Unit tests pass in a fresh Python 3.12 environment.
- [ ] A software license chosen by the authors has been added.

## Exclusions

- [ ] No `outputs/` or `logs/` directory is tracked.
- [ ] No CSV other than the original HELOC input is tracked.
- [ ] No generated metric, prediction, table, prompt, or response is tracked.
- [ ] No paper PDF, LaTeX source, review, or supplementary file is tracked.
- [ ] No PNG, JPEG, SVG, spreadsheet, or other generated media is tracked.
- [ ] No model checkpoint, downloaded weight, cache, container image, or archive
      is tracked.
- [ ] No scheduler submission script or scheduler log is tracked.
- [ ] No hostname, username, home directory, institution path, account, queue,
      partition, reservation, or job ID is present.
- [ ] No API key, access token, credential, email address, or private URL is
      present.

## Suggested audit commands

```bash
git status --short
git ls-files

git ls-files | grep -E \
  '\.(bin|ckpt|gguf|img|jpeg|jpg|onnx|pdf|png|pt|pth|safetensors|sif|svg|webp|xlsx)$' \
  && echo "Unexpected binary/media file found" || true

grep -RInE \
  '(SBATCH|SLURM|partition|qos|account|/home/|/Users/|lustre|api[_-]?key|access[_-]?token)' \
  --exclude-dir=.git --exclude='heloc_dataset_v1.csv' . \
  && echo "Review possible private infrastructure details" || true
```

Inspect every match; documentation may intentionally name a forbidden category
without containing a private value. Finally, create the public archive from
tracked files only rather than copying a working directory with ignored files.

