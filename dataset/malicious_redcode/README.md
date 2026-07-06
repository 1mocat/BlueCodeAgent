# malicious_redcode — gated dataset

Per the paper's **Impact Statement**, BlueCodeAgent's malicious-code instruction data is
**not distributed openly** (research-only access). The JSON files for this subset are
intentionally git-ignored and therefore absent from the public repository.

- **Derived from:** RedCode-Gen (Guo et al., 2024, *RedCode: Risky Code Execution and
  Generation Benchmark for Code Agents*).
- **Expected files (git-ignored):** `malicious_redcode_knowledge.json`, `malicious_redcode_test.json`
- **How to obtain the processed split:** request access from the authors at chengquanguo@uchicago.edu.

Once obtained, place the two files here and run `scripts/run_malicious.sh`.
