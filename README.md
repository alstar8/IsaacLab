# Installation

```bash
# Create and activate a uv virtual environment
cd ./projects/IsaacLab_release_3_0
uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate

uv pip install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
./isaaclab.sh --install
```

# Training command

```bash
source env_isaaclab/bin/activate
cd ./projects/IsaacLab_release_3_0

./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --headless physics=newton_mjwarp
```

[INFO] Logging experiment in directory: ./ReseachOS/IsaacLab_release_3_0/logs/rsl_rl/franka_deformable

# Eval
```bash
source env_isaaclab/bin/activate
cd ./projects/IsaacLab_release_3_0

./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --headless \
  --video --video_length 400 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-06-11_15-09-40/model_999.pt
```

# Export reward and success-rate curves from TensorBoard
```bash
source env_isaaclab/bin/activate
cd ./projects/IsaacLab_release_3_0

./isaaclab.sh -p scripts/reinforcement_learning/export_training_curves.py \
  --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-06-11_15-09-40/model_999.pt
```

# Eval with success-rate measurement (video + JSON summary)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/eval_checkpoint.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --headless --video --video_length 400 --num_envs 32 --num_steps 900 \
  --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-06-11_15-09-40/model_750.pt \
  physics=newton_mjwarp
```

# OAT (Ordered Action Tokenization) pipeline

Scripts in `scripts/reinforcement_learning/oat/` (vendored OAT core in `oat_tok/`, RSL-RL integration in `oat_rl/`).

```bash
# 1. Collect action chunks from a trained policy
./isaaclab.sh -p scripts/reinforcement_learning/oat/collect_actions.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 --headless \
  --num_envs 1024 --num_steps 640 --chunk_horizon 8 \
  --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-06-11_15-09-40/model_750.pt \
  --output datasets/oat/actions_model750_T8.pt physics=newton_mjwarp

# 2. Train the OAT tokenizer (no simulator needed)
python scripts/reinforcement_learning/oat/train_oat_tokenizer.py \
  --dataset datasets/oat/actions_model750_T8.pt \
  --output_dir logs/oat/tokenizer_T8_K4 --epochs 50

# 3. Train PPO with discrete OAT-token actions
./isaaclab.sh -p scripts/reinforcement_learning/oat/train_oat_ppo.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 --headless \
  --num_envs 4096 --max_iterations 400 \
  --tokenizer logs/oat/tokenizer_T8_K4/oat_tokenizer.pt \
  --experiment_name dexsuite_kuka_allegro_oat physics=newton_mjwarp

# 4. Evaluate the OAT-token policy (video + success rate)
./isaaclab.sh -p scripts/reinforcement_learning/oat/eval_oat_checkpoint.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 --headless \
  --video --video_length 400 --num_envs 32 --num_steps 120 \
  --tokenizer logs/oat/tokenizer_T8_K4/oat_tokenizer.pt \
  --checkpoint logs/rsl_rl/dexsuite_kuka_allegro_oat/2026-06-11_22-59-11/model_399.pt \
  physics=newton_mjwarp
```

# OAT: BC warm-start + anchored PPO (Idea C)

```bash
# Per-token salience of a tokenizer (tests the "safe tail tokens" premise, no simulator)
python scripts/reinforcement_learning/oat/token_salience.py \
  --tokenizer logs/oat/tok_T1_r16_c64/oat_tokenizer.pt \
  --dataset datasets/oat/actions_model750_T1.pt \
  --output logs/oat/tok_T1_r16_c64/token_salience.json

# BC warm-start + PPO with an annealed DAgger-style BC anchor
# (lambda(t) * CE(actor logits, expert tokens) protects the warm start from early-PPO collapse)
./isaaclab.sh -p scripts/reinforcement_learning/oat/bc_anchor_warmstart_ppo.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 --headless \
  --num_envs 4096 --max_iterations 1000 \
  --tokenizer logs/oat/tok_T1_r16_c64/oat_tokenizer.pt \
  --baseline_checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-06-11_15-09-40/model_750.pt \
  --bc_rollout_steps 128 --bc_epochs 80 --ppo_entropy_coef 0.002 \
  --bc_anchor_coef 0.3 --bc_anchor_iters 400 --bc_anchor_decay cosine \
  --experiment_name dexsuite_kuka_allegro_oat_anchor physics=newton_mjwarp
```