"""Script for running the webcam inference demo.

Usage:
    python infer.py
    python infer.py --camera 1
    python infer.py --confidence 0.7
    python infer.py --model ../models/xgboost_asl.pkl
    python infer.py --model ../models/neural_network_asl.pkl
"""

import argparse
from pathlib import Path

from src.config import config
from src.inference.webcam_demo import WebcamDemo


def main() -> None:
    """Runs the webcam inference demo."""
    parser = argparse.ArgumentParser(description="ASL classification demo")
    parser.add_argument("--model", type=str, default=None, help="Path to the .pkl model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--confidence", type=float, default=0.4, help="Minimum confidence")


    args = parser.parse_args()

    default_model_path = config.paths.models_dir / f"{config.training.model}_asl.pkl"
    model_path = Path(args.model) if args.model else default_model_path

    if not model_path.exists():
        print(f" Model not found: {model_path}")
        print("   Run first: python train.py")
        return

    print("=" * 60)
    print("ASL classification demo with webcam")
    print("=" * 60)
    print(f"\n Model: {model_path}")
    print(f" Camera: {args.camera}")
    print(f" Minimum confidence: {args.confidence:.0%}")

    demo = WebcamDemo(model_path=model_path, min_confidence=args.confidence)
    demo.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
