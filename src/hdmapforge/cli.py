"""Runtime entry points. Raw-data inspection and preprocessing are intentionally external."""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="HDMapForge hybrid geometry and VLM pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--raster", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--tile-size", type=int, default=512)
    predict.add_argument("--window", nargs=4, type=int, metavar=("COL", "ROW", "WIDTH", "HEIGHT"))
    vector = sub.add_parser("export")
    vector.add_argument("--probabilities", required=True)
    vector.add_argument("--raster", required=True)
    vector.add_argument("--output", required=True)
    vector.add_argument("--crs", required=True)
    vector.add_argument("--offset", type=float, nargs=2, default=[0, 0])
    vector.add_argument("--vlm-config")
    args = parser.parse_args()
    if args.command == "train":
        from .training import train as fit

        config_path = Path(args.config).resolve()
        print(
            fit(
                yaml.safe_load(config_path.read_text(encoding="utf-8")),
                config_dir=config_path.parent,
            )
        )
    elif args.command == "predict":
        from .inference import load_predictor
        from .pipeline import predict_raster

        print(
            predict_raster(
                args.raster,
                args.output,
                load_predictor(args.checkpoint, args.device),
                tile_size=args.tile_size,
                window=args.window,
            )
        )
    else:
        from .pipeline import vectorize_prediction
        from .vlm import VisionClassifier

        config = (
            yaml.safe_load(Path(args.vlm_config).read_text(encoding="utf-8"))
            if args.vlm_config
            else {}
        )
        runtime = config.get("runtime", {})
        classifier = VisionClassifier(**config["provider"]) if config.get("provider") else None
        try:
            result = vectorize_prediction(
                args.probabilities,
                args.raster,
                args.output,
                output_crs=args.crs,
                offset=args.offset,
                classifier=classifier,
                **runtime,
            )
            print(json.dumps({k: v for k, v in result.items() if k != "line_traces"}, indent=2))
        finally:
            if classifier:
                classifier.close()


if __name__ == "__main__":
    main()
