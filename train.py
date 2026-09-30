import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg, VICReg, VICRegLogDet
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def subset_episodes(dataset, frac, seed):
    """Subset holding every window of a random `frac` of the EPISODES."""
    eps = np.asarray([ep for ep, _ in dataset.clip_indices])
    uniq = np.unique(eps)
    n_keep = max(1, int(round(frac * len(uniq))))
    order = np.random.default_rng(seed).permutation(uniq)
    keep = np.isin(eps, order[:n_keep])
    idx = np.flatnonzero(keep).tolist()
    print(f"episode_frac={frac}: kept {n_keep}/{len(uniq)} episodes, "
          f"{len(idx):,}/{len(eps):,} windows", flush=True)
    return torch.utils.data.Subset(dataset, idx)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    loss_type = cfg.loss.get("type", "sigreg")
    if loss_type == "vicreg":
        loss, parts = self.vicreg(pred_emb, tgt_emb, ctx_emb, return_parts=True)
        output["loss"] = loss
        output["pred_loss"] = parts["sim"]   
        output["var_loss"] = parts["std"]    
        output["cov_loss"] = parts["cov"]
        pred_w = cfg.loss.vicreg.sim_w        
    elif loss_type == "vicreg_logdet":
        loss, parts = self.vicreg_logdet(pred_emb, tgt_emb, ctx_emb, return_parts=True)
        output["loss"] = loss
        output["pred_loss"] = parts["sim"]   
        output["ent_loss"] = parts["ent"]    
        pred_w = cfg.loss.vicreg_logdet.sim_w
    elif loss_type == "sigreg":
        # LeWM loss
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
        pred_w = 1.0
    else:
        raise ValueError(f"Unknown cfg.loss.type: '{loss_type}' "
                         f"(expected 'sigreg', 'vicreg' or 'vicreg_logdet')")
    
    beta = float(cfg.loss.get("beta", 0.0))
    delta = getattr(self.model, "_last_delta", None)
    if beta > 0 and delta is not None:
        output["disp_loss"] = beta * pred_w * delta.pow(2).mean()
        output["loss"] = output["loss"] + output["disp_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    ep_frac = float(cfg.get("episode_frac", 1.0))
    if ep_frac < 1.0:
        dataset = subset_episodes(dataset, ep_frac, seed=cfg.seed)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=True)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    opt_cfg = dict(cfg.optimizer)
    pred_mult = float(opt_cfg.pop("predictor_lr_mult", 1.0))
    sched = {"type": "LinearWarmupCosineAnnealingLR"}

    if pred_mult == 1.0:
        optimizers = {
            'model_opt': {
                "modules": 'model',
                "optimizer": opt_cfg,
                "scheduler": sched,
                "interval": "epoch",
            },
        }
    else:
        pred_cfg = dict(opt_cfg)
        pred_cfg["lr"] = float(opt_cfg["lr"]) * pred_mult
        optimizers = {
            'predictor_opt': {
                "modules": r'model\.(predictor|pred_proj)',
                "optimizer": pred_cfg,
                "scheduler": sched,
                "interval": "epoch",
            },
            'model_opt': {
                "modules": 'model',
                "optimizer": opt_cfg,
                "scheduler": sched,
                "interval": "epoch",
            },
        }
        print(f"[timescale] predictor lr {pred_cfg['lr']:.3e} "
              f"= {pred_mult}x encoder lr {float(opt_cfg['lr']):.3e}", flush=True)

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        vicreg = VICReg(**cfg.loss.get("vicreg", {})),
        vicreg_logdet = VICRegLogDet(**cfg.loss.get("vicreg_logdet", {})),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
