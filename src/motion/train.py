from Gesture_trainer import Gesture_trainer, Controlnet_trainer, MixTask_trainer
import argparse


def parse_args_for_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_yaml", default="", help="yaml path for training args")
    parser.add_argument("--controlnet", action='store_true', help="use controlnet or not")
    parser.add_argument("--mix_dataset", action='store_true', help="whether to mix datasets during training")
    args = parser.parse_args()
    return args


def train(args):
    yaml_file_path = args.train_args_yaml
    if args.controlnet:
        if args.mix_dataset:
            raise NotImplementedError("Mixing datasets with ControlNet is not implemented yet.")
        else:
            trainer = Controlnet_trainer(yaml_file_path=yaml_file_path)
    else:
        if args.mix_dataset:
            trainer = MixTask_trainer(yaml_file_path=yaml_file_path)
        else:
            trainer = Gesture_trainer(yaml_file_path=yaml_file_path)
    trainer.train_loop()


if __name__ == "__main__":
    args = parse_args_for_training()
    train(args)

