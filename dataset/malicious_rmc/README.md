# malicious_rmc — gated dataset

Per the paper's **Impact Statement**, BlueCodeAgent's malicious-code instruction data is
**not distributed openly** (research-only access). The JSON files for this subset are
intentionally git-ignored and therefore absent from the public repository.

- **Derived from:** RMCBench (Chen et al., 2024) — https://huggingface.co/datasets/zhongqy/RMCBench
- **Expected files (git-ignored):** `malicious_rmc_knowledge.json`, `malicious_rmc_test.json`
- **How to obtain the processed split:** request access from the authors at chengquanguo@uchicago.edu.

Once obtained, place the two files here and run `scripts/run_malicious.sh`.
