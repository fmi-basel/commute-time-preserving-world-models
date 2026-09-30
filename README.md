# CTWM — Commute-Time-Preserving World Models

Code for *Learning Commute-Time-Preserving World Models for Planning*.

This repository is based on [LeWM](https://github.com/lucas-maes/le-wm); the
contribution is confined to the objective (`module.py`) and the residual
predictor (`jepa.py`).

## Installation

Built on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
for environments, planning and evaluation, and
[stable-pretraining](https://github.com/galilai-group/stable-pretraining) for
training.

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets and checkpoints are read from `$STABLEWM_HOME`, which defaults to
`./data/` next to the scripts. You can override this path:

```bash
export STABLEWM_HOME=/path/to/your/storage
```

Download datasets from:

- **PushT, Two-Room, Cube and Reacher** &mdash; the datasets released with LeWM,
  from [HuggingFace](https://huggingface.co/collections/quentinll/lewm)
- **PointMaze** &mdash; the dataset can be similarly loaded from [HuggingFace]()
- **Scene** &mdash; dataset can be found
  [here](https://huggingface.co/datasets/galilai-group/ogb_scene_single/tree/main)
  in lance format.

The first two come as archives; decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted files under `$STABLEWM_HOME/datasets/`. For scene:

```bash
hf download galilai-group/ogb_scene_single --repo-type dataset \
    --local-dir $STABLEWM_HOME/datasets/ogbench
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.


## Training

`jepa.py` holds the model and `module.py` the objectives. Runs are configured
with [Hydra](https://hydra.cc/) under `config/train/`:

```bash
python train.py data=pusht
```

Checkpoints and the resolved config are written to
`$STABLEWM_HOME/checkpoints/<output_model_name>/`, one file per epoch.

To run: 

```bash
python train.py data=pusht
```

## Planning and evaluation

Evaluation configs live under `config/eval/`. `policy` is a checkpoint path relative to `$STABLEWM_HOME/checkpoints`:

```bash
python eval.py --config-name=pusht policy=<run_name>/weights_epoch_10.pt
python eval.py --config-name=pusht policy=random
```

For the hard setting (goal defined 100 steps ahead):

```bash
python eval.py --config-name=pusht policy=<run_name>/weights_epoch_10.pt \
    eval.goal_offset_steps=100 eval.eval_budget=125
```
