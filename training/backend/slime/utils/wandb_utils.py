import logging
import os
import re
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)

RUN_ID_FILE_NAME = "wandb_run_id.txt"


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _sanitize_run_id(name: str) -> str:
    # W&B run ids may only contain alphanumerics, dashes, underscores and dots,
    # and are capped at 128 characters.
    return re.sub(r"[^a-zA-Z0-9_.\-]", "-", name)[:128]


def _get_resume_run_id(args) -> str | None:
    """Run id of a previous wandb run to resume, or None to start a fresh run."""
    if getattr(args, "wandb_run_id", None):
        logger.info(f"Resuming wandb run id {args.wandb_run_id} from --wandb-run-id.")
        return args.wandb_run_id

    # Only auto-resume wandb when training itself is resuming from a checkpoint.
    load_dir = getattr(args, "load", None)
    if not (load_dir and os.path.exists(os.path.join(load_dir, "latest_checkpointed_iteration.txt"))):
        return None

    for ckpt_dir in (load_dir, getattr(args, "save", None)):
        if not ckpt_dir:
            continue
        run_id_file = os.path.join(ckpt_dir, RUN_ID_FILE_NAME)
        if os.path.exists(run_id_file):
            with open(run_id_file) as f:
                run_id = f.read().strip()
            if run_id:
                logger.info(f"Resuming wandb run id {run_id} recorded in {run_id_file}.")
                return run_id
    return None


def _persist_run_id(args, run_id: str):
    """Record the wandb run id in the save dir so a restarted job resumes this run."""
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, RUN_ID_FILE_NAME), "w") as f:
            f.write(run_id)
    except OSError:
        logger.warning(f"Failed to persist wandb run id to {save_dir}; wandb auto-resume on restart is disabled.")


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    resume_run_id = _get_resume_run_id(args)

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_config_for_logging(args),
        "resume": "allow",
    }

    if resume_run_id is not None:
        # Rejoin the previous run: wandb restores its name/group from the server,
        # so don't pass them here (a fresh random-suffix name would rename the run).
        init_kwargs["id"] = resume_run_id
    else:
        # add random 6 length string with characters
        if args.wandb_random_suffix:
            group = args.wandb_group + "_" + wandb.util.generate_id()
            run_name = f"{group}-RANK_{args.rank}"
        else:
            group = args.wandb_group
            run_name = args.wandb_group
            # Stable run name -> stable run id, so restarting the same experiment
            # resumes its wandb run even without a persisted run-id file.
            if run_name:
                init_kwargs["id"] = _sanitize_run_id(run_name)
        init_kwargs["group"] = group
        init_kwargs["name"] = run_name

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id
    # Record the run id next to the checkpoints so a restarted (crashed/stopped)
    # job automatically resumes logging into the same wandb run.
    _persist_run_id(args, wandb.run.id)
    logger.info(
        f"wandb initialized: project={args.wandb_project}, name={wandb.run.name}, "
        f"id={wandb.run.id}, resumed={bool(getattr(wandb.run, 'resumed', False))}"
    )


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    return deepcopy(args.__dict__)


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
