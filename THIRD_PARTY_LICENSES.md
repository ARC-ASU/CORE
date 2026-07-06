# Third-Party Licenses

Code and data authored for CORE are released under the MIT License (see [LICENSE](LICENSE)).
The repository bundles the following third-party components, which keep their original licenses:

| Component | Path | Upstream | License |
|:---|:---|:---|:---|
| verl (HybridFlow) | `verl/`, `scripts/model_merger.py` | https://github.com/volcengine/verl | Apache-2.0 (see [`verl/LICENSE`](verl/LICENSE)) |
| Qwen2.5-Math evaluation harness | `evaluation/` | https://github.com/QwenLM/Qwen2.5-Math | Apache-2.0 (see [`evaluation/LICENSE`](evaluation/LICENSE)) |
| latex2sympy2 | `evaluation/latex2sympy/` | https://github.com/OrangeX4/latex2sympy | MIT |
| LLaMA-Factory | `fine-tune/LLaMA-Factory/` | https://github.com/hiyouga/LLaMA-Factory | Apache-2.0 (see [`fine-tune/LLaMA-Factory/LICENSE`](fine-tune/LLaMA-Factory/LICENSE)) |

The `verl/` and `evaluation/` trees contain modifications made for CORE (concept-guided
rollout replacement, KL alignment, and evaluation-harness fixes). Original copyright
headers are retained where present.
