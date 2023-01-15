import argparse
from simplenmt.training import train
from simplenmt.prediction import test

def main():
    ap = argparse.ArgumentParser("simpleNMT")

    ap.add_argument("mode", choices=["train", "test"])

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("-c", "--ckpt", type=str, default="./checkpoints")

    ap.add_argument("-o", "--output_path", type=str, default="./checkpoints")

    args = ap.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        test(
            cfg_file=args.config_path,
            ckpt=args.ckpt,
            output_path=args.output_path
        )
    else:
        raise ValueError("Unknown mode")
    
if __name__ == "__main__":
    main()