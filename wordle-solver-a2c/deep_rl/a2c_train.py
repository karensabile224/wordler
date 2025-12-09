"""Advantage Actor Critic (A2C)"""

from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import torch

sys.path.insert(0, "/content/wordle-solver")
sys.path.insert(0, "/content/wordle-solver/deep_rl")

# enable A100 Tensor Core optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from a2c.module import AdvantageActorCritic


def cli_main() -> None:
    parser = ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = AdvantageActorCritic.add_model_specific_args(parser)

    # add resume support
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # opptional wandb logging bc colab doesn't seem to like wandb
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(project="wordle-solver")
            wandb.config.update(args)
        except:
            print("Warning: wandb not available, continuing without it")

    # find checkpoint to resume from if --resume flag is set
    ckpt_path = None
    if args.resume:
        import glob
        import os

        checkpoints = glob.glob("checkpoints/*.ckpt")
        if checkpoints:
            ckpt_path = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from: {ckpt_path}")
        else:
            print("No checkpoints found, starting fresh")

    # create or load model
    if ckpt_path:
        model = AdvantageActorCritic.load_from_checkpoint(ckpt_path, **vars(args))
    else:
        model = AdvantageActorCritic(**args.__dict__)

    # save checkpoints
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=100,
        save_last=True,  # always save last checkpoint
        dirpath="checkpoints/",
        filename="a2c-{epoch:02d}",
    )

    seed_everything(123)

    trainer = Trainer.from_argparse_args(
        args, deterministic=True, callbacks=[checkpoint_callback]
    )

    trainer.fit(model, ckpt_path=ckpt_path if ckpt_path else None)


if __name__ == "__main__":
    cli_main()
