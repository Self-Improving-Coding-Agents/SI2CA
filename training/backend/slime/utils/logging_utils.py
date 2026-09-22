import logging

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str, *, commit: bool = True):
    step = metrics.get(step_key)

    if args.use_wandb:
        # Pin wandb's global `_step` to the meaningful iteration step (the value under
        # ``step_key``) so every panel shares one consistent x-axis. Without an explicit
        # ``step=``, wandb auto-increments its internal `_step` on *every* log() call
        # (several per rollout: rollout, train, eval, multi_turn, ...), which inflates the
        # default "Step" axis far past the real iteration count and makes panels that fall
        # back to it disagree with those that resolve the define_metric step_metric.
        # All step_key values (train/step, rollout/step, eval/step) live on the same
        # monotonic scale (see compute_rollout_step), so this stays non-decreasing.
        if step is not None:
            wandb.log(metrics, step=int(step), commit=commit)
        else:
            wandb.log(metrics, commit=commit)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
