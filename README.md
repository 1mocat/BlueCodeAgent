# BlueCodeAgent

Official implementation for **"BlueCodeAgent: A Blue Teaming Agent Powered by Automated Red Teaming for CodeGen AI"** (ICML 2026).

<p align="center">
  <img src="assets/overview.png" alt="BlueCodeAgent overview" width="100%">
  <br><sub><b>BlueCodeAgent overview.</b> Automated red teaming accumulates diverse risk knowledge that powers a knowledge-enhanced blue-teaming agent, yielding much cleaner safe/unsafe classification than baseline detectors.</sub>
</p>

BlueCodeAgent is an end-to-end **blue-teaming (defense)** agent for CodeGen AI security. It turns knowledge accumulated by **automated red teaming** into two complementary defenses: **principled-level** defense (summarizing retrieved knowledge into actionable *constitutions*) and **nuanced-level** analysis (dynamic sandbox testing), and uses them to decide whether an input is *safe* or *unsafe* across four risk categories:

| Task | What it detects |
|------|-----------------|
| **bias** | Bias/unfairness embedded in code instructions |
| **vulnerable code** | Code containing CWE vulnerabilities (with dynamic testing to cut false positives) |
| **prompt injection** | Prompt-injection attacks against coding tasks |
| **malicious** | Requests to generate malware (RMCBench- and RedCode-Gen-derived) |

## How it works

1. **Retrieval** — embed the test instance, retrieve the top-`k` (default 3) most similar entries from the knowledge base.
2. **Constitution (principled-level)** — a summarizer model (GPT-4o) turns the retrieved knowledge into concise, actionable constitutions that guide the base model's safe/unsafe decision.
3. **Dynamic testing (nuanced-level, vulnerable-code only)** — a static analyzer (guided by the constitution) → if a vulnerability is claimed, a dynamic analyzer generates executable tests → run them in an **isolated Docker sandbox** → a final analyzer combines static reasoning, generated tests, execution results, and the constitution.

## Case study

<p align="center">
  <img src="assets/case_study.png" alt="BlueCodeAgent case study" width="100%">
  <br><sub>On a biased coding request, direct prompting and both safety-reminder baselines still comply, while BlueCodeAgent distills constitutions from red-teaming knowledge and refuses — a principled defense.</sub>
</p>

## Repository layout

```
BlueCodeAgent/
├── dataset/                     # released knowledge + test data (already embedded)
│   ├── bias/ vulnerability/ prompt_injection/     # public
│   └── malicious_rmc/ malicious_redcode/          # GATED (see their README.md)
├── src/
│   ├── blue_team/               # the four detectors + embedding utilities
│   ├── models/client.py         # LLM client wrappers (keys read from env)
│   ├── utils/                   # Docker sandbox + I/O helpers
│   └── evaluation/metrics.py    # unified F1 / TP-FP-TN-FN scoring
├── scripts/                     # one runnable entry point per task
├── environment/                 # Dockerfile + build context for the sandbox
├── requirements.txt  ·  .env.example
```

## Setup

```bash
# 1. Install dependencies (pip)
pip install -r requirements.txt

# 2. Provide API keys
cp .env.example .env      # then edit .env (never commit it)

# 3. Docker (only needed for the vulnerable-code dynamic testing)
#    A running Docker daemon is required; the image `red-teaming-code-agent`
#    is auto-built from environment/ on first use.
```

Keys used: **OpenAI** (GPT-4o + embeddings) and **Anthropic** (Claude dynamic analyzer). **Together** (open-source base models) is optional — only needed if you route base models through it.

## Data

Each task ships a **knowledge** file (retrieval base) and a **test** file, both pre-embedded — see `dataset/README.md` for counts, splits, and provenance. Sources: SecCodePLT (vulnerable code), MBPP (benign coding tasks), RedCode-Gen / RMCBench (malicious).

> **Gated data.** Per the paper's Impact Statement, the two malicious subsets are **not distributed openly**. `dataset/malicious_rmc/` and `dataset/malicious_redcode/` ship with only a README and no data. Qualified researchers may request access for defensive research by emailing **chengquanguo@uchicago.edu**; access is granted in accordance with the source benchmarks' licenses.

## Run the experiments

```bash
# Each script defaults to model=gpt4o, constitution=gpt4o, topk=3.
./scripts/run_malicious.sh            # needs the gated malicious data
./scripts/run_bias.sh                 # -> results/bias/
./scripts/run_prompt_injection.sh     # -> results/prompt_injection/
./scripts/run_vulnerable_code.sh      # -> results/vulnerability/   (needs Docker)
```

Each detector records, per item, the four comparison responses (direct prompting, general safety reminder, fine-grained safety reminder, and BlueCodeAgent's constitution). Score F1 / TP-FP-TN-FN for every method with the unified evaluator (`metrics.py`):

```bash
# metrics.py scores result JSONs into TP/FP/TN/FN, Precision, Recall, and F1
# (convention: unsafe == positive). Point it at a results/ directory or a single file.
python src/evaluation/metrics.py --task bias             --result_dir results/bias
python src/evaluation/metrics.py --task prompt_injection --result_dir results/prompt_injection
python src/evaluation/metrics.py --task vulnerable       --result_file results/vulnerability/result.json
# add --csv out.csv to also save the table as CSV
```

Embeddings are already included in the released data. To embed **new** data:
`scripts/embed_prompt.sh <in> <out>` (text tasks) or `scripts/embed_code.sh <in> <out>` (vulnerable code).

## Red-teaming

For the automated red-teaming method (adaptive jailbreak/attack optimization), please refer to **RedCodeAgent** (Guo et al., 2025, *RedCodeAgent: Automatic red-teaming agent against diverse code agents*). This repository releases the **datasets produced after red teaming** and the blue-teaming defense.

## Responsible use

This project is intended for **defensive security research**. The malicious-code data is gated to researchers per the paper's Impact Statement; please use all materials responsibly and in accordance with the source benchmarks' licenses.

## Citation

```bibtex
@inproceedings{
guo2026bluecodeagent,
title={BlueCodeAgent: A Blue Teaming Agent Powered by Automated Red Teaming for CodeGen {AI}},
author={Chengquan Guo and Yuzhou Nie and Chulin Xie and Zinan Lin and Wenbo Guo and Bo Li},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=TR2DYfZXTd}
}
```
