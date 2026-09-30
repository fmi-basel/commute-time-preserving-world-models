import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import types
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def add_scene_button_adapters(world):

    def set_state_from_row(self, qpos, qvel, button_states):
        buttons = {f"button_state_{i}": int(v)
                   for i, v in enumerate(np.ravel(button_states))}
        self.set_state(qpos, qvel, **buttons)

    def set_target_buttons(self, button_states):
        for i, v in enumerate(np.ravel(button_states)):
            self.set_target_button_state(i, int(v))

    n = 0
    for env in getattr(world.envs, "envs", []):
        e = env.unwrapped
        if not hasattr(e, "set_target_button_state"):
            continue
        e.set_state_from_row = types.MethodType(set_state_from_row, e)
        e.set_target_buttons = types.MethodType(set_target_buttons, e)
        n += 1
    return n


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def episode_index_column(dataset):
    """Name of the per-row episode id column.

    Lance hides its configured index columns from `column_names` -- they are the
    reader's episode_index_column/step_index_column -- so a lance dataset that
    HAS episode_idx still reports it missing here and falls back to ep_idx.
    """
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def episode_ids(dataset):
    """Per-row episode ids, as a flat int array.

    Published lance datasets can carry ep_idx as (N, 1) float32 where HDF5 uses
    1-D int32. Left alone that breaks three separate things downstream: the
    `episode_idx == ep_id` mask becomes 2-D and cannot index a 1-D step_idx, the
    max_start_idx lookup tries to hash a (1,) array as a dict key, and float ids
    never compare equal to the int keys built from np.unique.
    """
    col = np.asarray(dataset.get_col_data(episode_index_column(dataset)))
    return col.reshape(-1).astype(np.int64)


def step_ids(dataset):
    """Per-row step ids, as a flat int array (see episode_ids)."""
    return np.asarray(dataset.get_col_data("step_idx")).reshape(-1).astype(np.int64)


def get_episodes_length(dataset, episodes):
    episode_idx = episode_ids(dataset)
    step_idx = step_ids(dataset)
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    h5_path = dataset_path / "datasets" / f"{dataset_name}.h5"
    if h5_path.exists():
        return swm.data.HDF5Dataset(
            dataset_name,
            keys_to_cache=cfg.dataset.keys_to_cache,
            cache_dir=dataset_path,
        )
    return swm.data.load_dataset(
        dataset_name,
        cache_dir=str(dataset_path),
        keys_to_cache=cfg.dataset.keys_to_cache,
    )

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget

    world = swm.World(**cfg.world, image_shape=(224, 224))

    # Maze goal marker: disable it, exactly as dataset_collection.py does before
    # rendering. In states mode OGBench adds a `target` cylinder to the maze XML
    # as a visualisation aid and set_goal() MOVES it to the episode's goal, so it
    # is rendered into the observations the encoder reads -- at a position that
    # changes per episode. The goal image, by contrast, is a frame from the
    # dataset, so the two disagree about where the marker is, and the planning
    # cost ||phi(z_H) - phi(goal)||^2 is computed across that difference. Turning
    # the geom transparent makes the eval env match the env the data was rendered
    # with. No pixels are touched; no-op for environments without the geom.
    for _env in getattr(world.envs, "envs", []):
        try:
            _env.unwrapped.model.geom("target").rgba[3] = 0.0
        except (AttributeError, KeyError, ValueError):
            pass

    n_btn = add_scene_button_adapters(world)
    if n_btn:
        print(f'scene: button-state adapters bound on {n_btn} envs')

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    ep_indices, _ = np.unique(episode_ids(stats_dataset), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Read once and reuse: on lance each of these is a full-column scan.
    ep_col = episode_ids(dataset)
    step_col = step_ids(dataset)
    # Map each dataset row’s episode_idx to its max_start_idx
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in ep_col])

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = step_col <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = ep_col[random_episode_indices]
    eval_start_idx = step_col[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    save_video = bool(cfg.eval.get("save_video", True))
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path if save_video else None,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
