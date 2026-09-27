# HELOC input data

`heloc_dataset_v1.csv` is the anonymized input released for the FICO
Explainable Machine Learning Challenge. It contains 10,459 records, the binary
`RiskPerformance` target, and 23 credit-report features.

Original challenge page:
<https://community.fico.com/s/explainable-machine-learning-challenge>

Integrity check:

```text
SHA-256  abdb86f415228b6754dcdee791feaf25bb73a9147f1742cf09c9a2dd54308a93
Rows     10,459 data rows (10,460 lines including the header)
```

The special values `-9`, `-8`, and `-7` are treated as missing/special values.
The preprocessing code imputes their numeric feature with the training-split
median and adds a paired missingness indicator. No derived data or experiment
result is stored in this directory.

If local redistribution rules require the CSV to be omitted, remove it before
publication and instruct users to download `heloc_dataset_v1.csv` from the
source, place it at this path, and verify the checksum above.

