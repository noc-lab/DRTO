# DRTO: Distributionally Robust Token Optimization in RLHF

Code for [Distributionally Robust Token Optimization in RLHF](https://arxiv.org/abs/2604.08577).
DRTO cuts each rollout into spans at low-confidence tokens and re-weights the span losses with a
KL or χ² ambiguity set, on top of the token-level reward of
[RTO](https://arxiv.org/abs/2404.18922).

## Setup

```bash
conda create -n drto python=3.10 -y && conda activate drto
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
pip install -e .
```

## Data

```bash
python scripts/prepare_math_data.py      # OpenMathInstruct-1 prompts -> data/openmath_100k.csv
```

## Reward model (optional)

Training uses the public `RTO-RL/Llama3.2-1B-RewardModel` by default. To train your own on UltraFeedback:

```bash
bash scripts/train_rm.sh                 # saves to ckpt/rm
RM=ckpt/rm bash scripts/train_chi_drto_span_math.sh
```

## Train DRTO (Llama3-8B, math)

```bash
bash scripts/train_chi_drto_span_math.sh   # χ²-DRTO
bash scripts/train_kl_drto_span_math.sh    # KL-DRTO
```

Models are saved merged to `ckpt/math/<method>/`. Runs on 2 GPUs by default
(`CUDA_VISIBLE_DEVICES` to change). Any setting can be overridden from the environment, e.g.
`RHO=0.2 MAX_SAMPLES=2500 bash scripts/train_chi_drto_span_math.sh`.

| Role | Model |
|---|---|
| Actor / reference | `OpenRLHF/Llama-3-8b-sft-mixture` |
| DPO policy (token reward) | `RTO-RL/Llama3-8B-DPO` |
| Reward model / critic init | `RTO-RL/Llama3.2-1B-RewardModel` |

## Main arguments

| Argument | Default | Meaning |
|---|---|---|
| `--span_dro_type` | `chi2` | `chi2` or `kl` |
| `--span_min_len` | 16 | minimum span length |
| `--span_max_num` | 8 | maximum spans per response |
| `--span_cut_prob` | 0.05 | cut-point probability threshold |
| `--span_chi_rho` | 0.1 | χ² radius |
| `--span_kl_rho` | 0.01 | KL radius |
| `--log_span_stats` | off | log span count / length per step |

## Citation

```bibtex
@article{jin2026drto,
  title   = {Distributionally Robust Token Optimization in RLHF},
  author  = {Jin, Yeping and Hu, Jiaming and Paschalidis, Ioannis Ch.},
  journal = {arXiv preprint arXiv:2604.08577},
  year    = {2026}
}
```

Built on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Apache-2.0 license.
