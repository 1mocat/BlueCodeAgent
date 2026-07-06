# BlueCodeAgent Datasets

Benchmark data for the four tasks addressed by BlueCodeAgent. Each task
directory holds a **knowledge** file (retrieval knowledge base) and a **test** file
(evaluation set). Every record carries precomputed embeddings (`prompt_embedding`,
and for `vulnerability` also `safe_embedding` / `unsafe_embedding` / `llm_code_embedding`)
so the retrieval pipeline can be reproduced without re-encoding.

## Layout

```
dataset/
├── bias/                 bias_knowledge.json              bias_test.json
├── vulnerability/        vulnerability_knowledge.json     vulnerability_test.json
├── prompt_injection/     prompt_injection_knowledge.json  prompt_injection_test.json
├── malicious_rmc/        malicious_rmc_knowledge.json     malicious_rmc_test.json
└── malicious_redcode/    malicious_redcode_knowledge.json malicious_redcode_test.json
```

Each JSON file is a list of records. Load, e.g.:

```python
import json
knowledge = json.load(open("dataset/bias/bias_knowledge.json"))
test      = json.load(open("dataset/bias/bias_test.json"))
```

## Contents

| Task | Split | Knowledge (#) | Test (#) |
|------|-------|--------------:|---------:|
| `bias`              | ood | 528 | 504 |
| `vulnerability`     | ood | 260 | 280 |
| `prompt_injection`  | ood | 240 | 240 |
| `malicious_rmc`     | ood | 252 | 272 |
| `malicious_redcode` | ood | 193 | 199 |