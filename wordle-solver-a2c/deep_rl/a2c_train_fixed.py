from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.insert(0, '/content/wordle-solver')
sys.path.insert(0, '/content/wordle-solver/deep_rl')

from a2c.module import AdvantageActorCritic

def cli_main() -> None:
  parser = ArgumentParser(add_help=False)
  
  # Add model args
  parser = AdvantageActorCritic.add_model_specific_args(parser)
  
  # Add trainer args manually
  parser.add_argument("--max_epochs", type=int, default=10)
  parser.add_argument("--accelerator", type=str, default="cpu")
  parser.add_argument("--devices", type=int, default=1)
  parser.add_argument("--log_every_n_steps", type=int, default=50)
  parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
  
  args = parser.parse_args()
  
  # Initialize wandb if requested (for now, skip it to avoid errors)
  # import wandb
  # wandb.init(project='wordle-solver', config=vars(args))
  
  # Create model
  model = AdvantageActorCritic(**vars(args))
  
  # Save checkpoints
  checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=100,
    dirpath='checkpoints/',
    filename='a2c-{epoch:02d}'
  )
  
  seed_everything(123)
  
  # Create trainer with compatible args
  trainer = Trainer(
    max_epochs=args.max_epochs,
    accelerator=args.accelerator,
    devices=args.devices,
    log_every_n_steps=args.log_every_n_steps,
    deterministic=True,
    callbacks=[checkpoint_callback]
  )
  
  trainer.fit(model)

if __name__ == '__main__':
  cli_main()
