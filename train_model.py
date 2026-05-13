import argparse
import yaml
from src.trainer import Trainer


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Train a BraTS segmentation model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    # In case there is a need to stop training and resume later from a checkpoint (last trained epoch) 
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from (e.g. checkpoints/swin_unetr/last_model.pt).")
    parsed = parser.parse_args(args)

    with open(parsed.config) as f:
        config = yaml.safe_load(f)

    Trainer(config, resume_path=parsed.resume).fit()


if __name__ == "__main__":
    main()
