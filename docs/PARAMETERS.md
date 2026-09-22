# Parameter reference

## Public package CLI (v0.2)

`si2ca run --method ...` and `python -m si2ca.run --setting ...` run the selected experiment directly. All commands support `-h`/`--help`, and the root supports `--version`. `si2ca ls methods|settings|benchmarks` discovers available choices. `si2ca ls skills` prints the installed proposer and recorder skill paths without starting agents or experiments; see the [search-cycle guide](SEARCH.md).

| Public option | Meaning |
|---|---|
| `--method Standard|SJ|SL|Discovered|SFT` | Case-insensitive method; Discovered selects the Discovered Early-Commit Window Strategy. `--setting` selects a named recipe on the unified harness. |
| `--backend self-hosted|api` | Self-hosted uses SGLang capabilities; API is generation-only and cannot use SL. DeepSWE presets default to API; otherwise self-hosted. |
| `--model` | Served/API model ID with `--base-url`; otherwise local checkpoint or Hugging Face `org/model` to serve automatically. |
| `--model-path` | Optional local checkpoint directory while keeping the full HF `--model` ID. A complete directory is reused; missing files download here. |
| `--revision` | HF branch, tag or commit used for downloaded model/tokenizer files. Explicit existing local directories remain authoritative. |
| `--model-cache-dir` | HF cache root; otherwise use standard `HF_HOME` / `HF_HUB_CACHE` settings. Not the benchmark-assets cache. |
| `--local-files-only` | Forbid model/tokenizer downloads; missing or incomplete cache fails explicitly. |
| `--tool-call-parser` / `--reasoning-parser` | Managed SGLang parsers: `auto` (known-family defaults), `none`, or an explicit SGLang parser name. Unknown models are not forced into Qwen parsing. |
| `--gpu-ids` / `--gpus` / `--gpu` | Physical IDs for one managed replica. Reuses a matching server on `--port`, otherwise starts it. |
| `--tp` | Managed tensor parallelism; defaults to the number of listed GPU IDs. |
| `--port` | Managed local service port, default 8151. |
| `--context-length` / `--mem-fraction` / `--rocm` | Managed server context (default: infer from model), static memory fraction (.8), and AMD visibility flags. |
| `--keep-server` | Keep only a newly created server after the run. Reused services are always left running. |
| `--out` / `--output-dir` | Optional publicly; default is a timestamped directory in `runs/`. Use a stable explicit path to resume. |
| `--num-candidates` / `--branch` / `--k` | Candidates drawn per branch. SJ defaults to 2; SL defaults to 4. |
| `--judge-samples` / `--average` / `--score-samples` | Independent judge samples averaged for each candidate; SJ defaults to 3. |
| `--gold-patch` / `--no-gold-patch` | SJ judge sees the reference patch / uses intrinsic rubric only. Equivalent to SJ `--pi gold|none`. |
| `--gold-weight` | SJ G1 weight in [0,1], default .3; intrinsic weights are multiplied by `1-weight`. Also configurable for the discovered early-commit window strategy. |
| `--gold-field` | Explicit dotted row field; otherwise common gold/reference/patch/answer fields are detected. Conflicts and missing required PI fail before running. |
| `--branch-until-edit` | Discovered early-commit mutation-command window, default 3; must be positive. |
| `--pi` / `--placement` / `--score` | SL: gold or hint; head or tail; likelihood or gain. Defaults: hint/head/likelihood. |
| `--judge-backend` | Independent judge request compatibility. Defaults to API for an explicitly external judge, otherwise the policy backend. |
| `--api-sampling` | Include temperature/top-p for APIs that support them; omitted otherwise. |
| `--api-token-field` | `max_tokens` (default) or `max_completion_tokens`, matching the API contract. |
| `SI2CA_CACHE_DIR` | Writable extraction cache root. Package data are not modified, so wheel installs can be read-only. |

The one-command installer accepts `--venv`, `--python`, and `--profile api|full|serve`. `api` installs the shared API/SL clients; `full` also installs Multilingual grading; `serve` additionally resolves the NVIDIA SGLang extra. ROCm kernels should come from the compatible platform image.

Both entry points use the [unified execution protocol](HARNESS.md). Named recipes retain explicit benchmark/method budgets; they do not reproduce every historical harness behavior.

Configuration precedence is shared defaults → named setting → explicitly supplied CLI values. `python -m si2ca.run ...` validates the configuration and inputs, then runs the experiment. YAML keys use underscores; equivalent CLI flags use hyphens. Paths in the bundled manifests resolve relative to the repository; custom relative asset paths resolve relative to the custom JSONL file.

## Experiment inputs and execution

| Parameter | Default | Meaning |
|---|---|---|
| `--setting` | Required | Name in `configs/experiments.yaml`. Determines the selector and budget on the shared harness. |
| `--config` | Bundled YAML | Alternative registry; edit a copy for recipe ablations. |
| `--benchmark` | `verified`; DeepSWE settings use `deepswe` | `verified`=500, `pro`=731 (full public split: Python, Go, JavaScript and TypeScript), `both`=1,231, `deepswe`=113, `validation`=192 SWE-bench Multilingual tasks. |
| `--dataset` | Bundled manifest | Custom task JSONL; replaces benchmark filtering. |
| `--validated-only` | False | Select only tasks explicitly marked `gold_validated=True` in the dataset. A subset is not a full search cycle. |
| `--allow-unvalidated` | False | Explicitly permit execution while some search grading chains remain unvalidated; does not certify the results. |
| `--model` | Required | Policy model name advertised by the service, not a checkpoint filesystem path. |
| `--base-url` | Required | Policy endpoint root or `/v1`; comma-separated replicas are assigned by task. |
| `--tokenizer-path` | SL: defaults to `--model-path` or `--model` | Exact policy tokenizer directory or HF ID used to re-render candidates and align token spans. Existing endpoints need only tokenizer/config downloads, never weights. |
| `--out` | Required | Isolated output directory. Different effective configurations cannot reuse it. |
| `--limit` | 0 | Positive value selects a debugging prefix; 0 uses all selected tasks. |
| `--concurrency` | 8 | Concurrent task sandboxes, not GPU count or candidate count. |
| `--seed` | 42 | Harness selection/shuffling seed. It does not guarantee deterministic remote sampling. |
| `--n-samples` | 1; Terra=4 | Independent trajectories per task, stored with a `sample` index in the shared result file. |
| `--task-timeout` | 100000000 s; DeepSWE=200000 s | Large finite task budget approximating the recorded no-time-limit protocol. |
| `--eval-timeout` | 3600 s | Verified/Pro/Multilingual grading timeout. DeepSWE uses its per-task verifier timeout, normally 1800 s plus transport allowance. |
| `engine` | `strategy` | The only public execution route; legacy engine overrides are rejected. |
| `strategy` | Per setting | baseline, selfguide_rubrics, sl, sl_gain or discovered. The low-level runner also accepts a reviewed Python strategy file. |

## Generation and guidance

| Parameter | Default | Meaning |
|---|---|---|
| `--temperature` | 1.0; strong judge ablation=0.6 | Policy sampling temperature. |
| `--top-p` | 0.95 | Nucleus probability mass. |
| `--top-k` | 10; strong judge ablation=20 | Policy top-k cutoff; not the number of candidate actions. |
| `--max-gen-tokens` | 4096; DeepSWE=0 | Maximum new policy tokens per draw; 0 defers to the API/shim cap. |
| `--max-turns` | 250; DeepSWE=1000 | Maximum policy/environment turns per trajectory. |
| `--reasoning-effort` | Endpoint default; DeepSWE=xhigh | API reasoning effort when supported. Local Qwen generation uses its configured reasoning parser. |
| `keep_reasoning` | True | Keep selected policy reasoning in live history; judge prefix rendering still removes previous reasoning. |
| `--k` | Standard=1, SJ=2, SL=4 | Candidates drawn per branched turn. Discovered strategy draws one outside its early window. |
| `--score-samples` | SJ=3; strong judge/DeepSWE=2 | Independent judge samples per candidate. Valid weighted totals are averaged. |
| `pi` | Per setting | none, gold or hint. Never appended to the policy generation context. |
| `position` | head or tail | Privileged block placement for SL rescoring only. |
| `--judge-url` | Policy chat URL | Full judge `/v1/chat/completions` URL; supported by the shared runner. |
| `--judge-model` | Policy model | Judge identity; mandatory explicit override for `sj_strong`. |
| `--judge-mode` | local_openai | OpenAI-compatible chat service. Legacy direct transport modes are rejected; use the Responses shim when needed. |
| `--judge-reasoning-effort` | medium; DeepSWE SJ=xhigh | Reasoning effort for the judge; use a matching effort-specific shim when applicable. |
| `--judge-temperature` / `--judge-top-p` | 1.0 / 0.95 | Explicit judge sampling; API capability filtering still applies. |
| `--judge-max-tokens` | 4096 | Judge output cap, separate from policy output. |
| `judge_thinking` | True | False sends `chat_template_kwargs.enable_thinking=False` for supported local judges. |
| `judge_batch_samples` | False | Independent requests for every score sample; true is rejected by the unified runner. |
| `prefix_head_chars` / `prefix_tail_chars` | 1500 / 1500 | Head and tail of each tool observation included in the judge prefix. |
| `prefix_max_chars` | 80000 | Judge trajectory budget; oldest blocks are removed first. The gold block is separate. |
| `--gold-max-chars` | 100000000; DeepSWE=16000 | Judge gold-patch truncation bound. Main Qwen SJ effectively keeps the full patch. |
| `rubric_family` | False | Keep false; historical action-family overrides are not supported by the unified general rubric. |

All new runs use the same configurable prefix renderer, tool execution and submission rules. DeepSWE keeps in-place grading but no separate policy loop. The API adapter omits unsupported local/sampling fields; `--api-sampling` enables temperature/top-p for compatible providers.

### Rubric weights

| Item | Meaning | Without PI | With PI |
|---|---|---:|---:|
| R1 | Reasoning grounded in observed evidence | .15 | .105 |
| R2 | Diagnostic insight / narrowing the cause | .15 | .105 |
| R3 | Coherent, nonredundant plan | .10 | .070 |
| A1 | Action implements the reasoning | .15 | .105 |
| A2 | Correct, specific command | .20 | .140 |
| A3 | Expected progress / information gain | .15 | .105 |
| A4 | Efficiency and safety | .10 | .070 |
| G1 | Convergence toward the reference solution | — | .300 |

Items are scored 1–10. The code computes the weighted sum; this has the same ranking as an affine normalization to [0,1]. Full SJ and Discovered share this scorer, skip identical command batches without null-control requests, and use seeded random selection for identical/tied candidates. SL selectors use the same identical/tie convention. See [historical differences](HARNESS.md#historical-paper-results).

SL takes the lowest privileged mean NLL; gain takes the highest plain-minus-privileged mean NLL. Missing scores have separate fallback decisions. `RESCORE_CONCURRENCY` defaults to 4 and limits simultaneous native rescoring forwards to control memory peaks. Token-span lengths above the retained 4096-span guard are skipped and marked unavailable.

## Authentication and transport

| Variable / option | Meaning |
|---|---|
| `SI2CA_API_KEY` | Policy chat/native bearer token; never stored in run config. |
| `SI2CA_JUDGE_API_KEY` | External chat judge token; falls back to policy token. |
| `COPILOT_TOKEN` / `GH_BIN` | Shim token override or GitHub CLI executable; otherwise shim uses `gh auth token`. |
| `COPILOT_BASE_URL` | Override the shim's remote base URL, including for offline mock servers. |
| `SHIM_USE_TRAPI` | Optional historical fallback channel; needs `TRAPI_BASE_URL`, `TRAPI_DEPLOYMENTS` (JSON, model to deployment) and `TRAPI_TOKEN_RESOURCE`. Leave unset for the documented Copilot-only setup. |
| Shim `--model`, `--effort` | Fixed upstream model and reasoning effort. |
| Shim `--host`, `--port` | Local bind address, default 127.0.0.1:8901. |
| Shim `--max-output-tokens` | Upstream output cap; generation example 65536, judge 4096. |
| Shim `--max-inflight` | Upstream concurrent requests; example 16. |
| Shim `--probe` | Performs real API effort probes; it is not an offline check. |

## Serving

`si2ca.serve` starts the service directly: `--model-path` selects local weights or a full HF ID; `--name` defaults to that unshortened value; `--gpus` lists physical IDs; `--tp` must equal that list's length; `--port` selects the service port; `--context-length` defaults to the model configuration; `--mem-fraction` defaults to .8; `--rocm` sets the recorded ROCm flags; `--log` must be a new log path. Model caching, `--revision`, `--model-cache-dir`, `--local-files-only`, `--tool-call-parser` and `--reasoning-parser` work as above. Qwen3.5 defaults to `qwen3_coder`/`qwen3`; other architectures are not assigned these parsers unconditionally. No inference-model allowlist is imposed; model, hardware and tool-call compatibility remain SGLang requirements.

## Data formats

Task JSONL rows contain `prompt`, `label`, and `metadata`. Required metadata: `instance_id`, `image`, `workdir`, `problem_statement`, plus a grading path. Verified/training tasks use `eval_cmd` and optional `eval_files`; Pro uses `swepro` setup, scripts and F2P/P2P lists; Multilingual uses `benchmark=swebench_multilingual`, `eval_assets.eval_body`, `log_parser`, `f2p`, `p2p`; DeepSWE uses `deepswe` verifier settings and `eval_files`. SJ requires `gold_patch`; hint-based SL requires `hint` for every selected task. Missing PI fails before running.

Bundled assets are addressed by relative paths and deduplicated by content. Search `gold_validated` records successful gold-patch grading, not model accuracy; see the per-task CSV and [provenance](provenance.json). Training inputs must be decontaminated from all evaluation tasks.

## SFT parameters

Pool construction: `--input-root` contains `pool_v1/swe_rl_9b_mix_v1.jsonl`, `swe_distill_5358.jsonl`, and decontaminated `swe_rl_python_all_v1.jsonl`; optional self-guided/teacher evidence uses the paths documented in the script. `--reward-files` accepts explicit evidence globs. `--out-dir` receives candidate/tier manifests. Tasks without execution evidence require environment validation; optional `training/validate_env.py --help` exposes the original checker.

Trajectory preparation: `--traj-dir` selects exported policy trajectories; `--tokenizer` selects the student tokenizer; `--out` must not exist; `--max-length` defaults to 131072; `--task-ids` fixes a shared task-ID subset. The audit reports rejected, retained and missing requested IDs. Solved and unsolved clean exits are both retained.

| SFT parameter | Value | Meaning |
|---|---|---|
| `SFT_DATA` | Required | Clean messages/tools JSONL for this arm. |
| `STUDENT_HF` / `STUDENT_REF` | Required | Base HF checkpoint / converted Megatron checkpoint. |
| `SAVE_DIR` | Required | Per-arm checkpoint output. |
| `MEGATRON_PATH` | Required | Compatible external Megatron-LM checkout. |
| `RESUME` | 0 | Explicitly load an existing checkpoint; otherwise start a new training run. |
| actor nodes / GPUs | 1 / 8 | Single eight-GPU worker node. |
| TP / SP / PP / CP | 2 / on / 1 / 4 | Tensor, sequence, pipeline and context parallelism; DP=1. |
| EP / expert TP | 8 / 1 | Expert parallelism and tensor parallelism within each expert. |
| epochs / global batch | 2 / 32 | Two passes; 32 trajectories per optimizer step. |
| rollout batch / samples | 32 / 1 | SFT data delivery batch; one trajectory per input. No online generation in SFT. |
| input / tool key | messages / tools | JSONL fields passed to the chat template. |
| shuffle / rollout seed | on / 42 | Shuffle training data reproducibly. |
| loss / mask | sft_loss / qwen3_5 | Assistant content and reasoning contribute; system/user/tool tokens are masked. |
| per-token loss | on | Token-normalized supervised objective. |
| disable advantages / debug-train-only | on / on | Skip RL advantages and rollout model serving. |
| sequence / max positions | 131072 / 131072 | Context bound; cleaner drops rather than truncates longer examples. |
| dynamic batch / max tokens per GPU | on / 32768 | Token-budget packing; 131072/CP4 fits the longest example. |
| recomputation | full / uniform / 1 layer | Recompute each layer to reduce activation memory. |
| log-probs chunk size | 4096 | Chunk logprob intermediates; does not eliminate output-logit allocation. |
| optimizer | AdamW | `--optimizer adam` with decoupled weight decay in this framework. |
| LR / minimum / schedule | 5e-6 / 5e-7 / cosine | Initial LR and cosine-decay floor. |
| warmup / weight decay | .03 / .1 | First 3% of steps warm up; constant decay coefficient .1. |
| Adam beta1 / beta2 / epsilon | .9 / .95 / 1e-8 | Optimizer moments and numerical stabilizer. |
| gradient clip | 1.0 | Global gradient-norm threshold. |
| distributed optimizer | on | Uses framework optimizer distribution; DP=1 has no DP sharding gain. |
| dropout attention / hidden | 0 / 0 | No training dropout. |
| precision | bf16, FP32 gradient accumulation and attention softmax | Recorded numerical recipe. |
| save interval | 20 | Checkpoints every 20 steps plus framework epoch/final boundaries. No automatic retention deletion. |

`training/model.sh` holds checkpoint architecture, not tunable experimental axes: 40 layers; hidden width 2048; 16 attention heads / 2 KV groups; KV channels 256; gated attention; QK normalization; no linear biases; RMSNorm epsilon 1e-6 with 1+p weights; RoPE base 1e7 and rotary fraction .25; SwiGLU; untied embeddings/output; vocabulary 248320; 256 experts on all layers, top-8 routing, expert/shared FFN width 512, softmax router in FP32, all-to-all dispatch, grouped GEMM, permutation fusion, probability-based drop policy, auxiliary loss 0, shared-expert/output gates. The Qwen3.5 model spec reads the HF layer types to select full versus linear attention.

Conversion: `--hf-checkpoint`/`--save` identify HF input and Megatron output; `--use-cpu-initialization` initializes on CPU but does not certify that all conversion work is GPU-free; `--bf16` selects weight precision. Reverse conversion uses `--input-dir` for a concrete iteration, `--output-dir` for new HF output, `--origin-hf-dir` for tokenizer/config metadata and `--add-missing-from-origin-hf` for absent nontrained weights. Optional converter `--model-name`, `--vocab-size`, `--chunk-size` control mapping, vocabulary unpadding and output shard size; avoid `--force` unless overwrite is intended.

### Environments

Recorded SFT runtime: Megatron-LM commit prefix `1dcf0dafa` (mcore 0.16.0rc0), torch 2.9.0a0+git7bcbafe / HIP 7.0.51831, TransformerEngine 2.10.0+b2a04508, transformers 5.6.0, Ray 2.44.1, mbridge 0.15.1. This is an environment record, not a claim that these GPU builds are installable from generic PyPI. External dependencies also include the compatible flash-linear-attention kernels and platform libraries used by the model spec.

The retained Ray training modules also import SGLang and sglang-router even in training-only mode. The original MI300 configuration names image `sglang:v0.5.11-rocm700-mi30x-patched`; the complete original patched image/environment or a compatible reconstruction is required. The CPU harness environment is not sufficient for training.

The launcher keeps the recorded ROCm scratch/kernel flags, GC threshold .8, explicit HIP/ROCR visibility, Ray no-reset visibility flags, disabled IB, TCP interface selection, asynchronous NCCL error handling, FP32 accumulation and Triton ROCm patch. `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` default to eth0 and can be changed to the actual interface. Do not mix the serving and training dependency sets.

## Search and result aggregation

`si2ca.search`: `--incumbent` and `--candidate` must cover the same complete 192 IDs; `--delta=7` is the solved-count gain; `--epsilon=2` is tolerated regression; `--ratio=.9` is the retained-cost ratio; `--context-audit-passed` records a completed human review of PI/task-identity restrictions. Missing retained-token accounting means the rule uses turns only; all-candidate generation tokens are never substituted.

`si2ca.results`: input paths are JSON/JSONL result files; `--expected` is the fixed number of task/sample trials, not the number of successful rows. Resolve threshold is reward ≥.999; turn P90 uses linear interpolation. New repeated DeepSWE trials share one result file with `sample` indices; group by sample and report their mean rate and sample standard deviation, not pass@4. Infrastructure failures keep the summary provisional (`final: false`) and are retried on resume; the denominator is not reduced. Turn statistics cover completed trials only (`turns_trials`). Old or mixed-version results cannot be resumed as unified runs.
