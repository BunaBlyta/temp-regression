import argparse
from pathlib import Path

import torch


MODEL_PATH = Path("temperature_model.pt")


def load_model():
    checkpoint = torch.load(MODEL_PATH)

    model = torch.nn.Linear(1, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["fahrenheit_scale"], checkpoint["celsius_scale"]


def main():
    parser = argparse.ArgumentParser(description="Convert Fahrenheit to Celsius with a trained PyTorch model.")
    parser.add_argument("fahrenheit", type=float, help="Temperature in Fahrenheit")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("temperature_model.pt not found. Run `python train.py` first.")

    model, fahrenheit_scale, celsius_scale = load_model()

    fahrenheit_tensor = torch.tensor([[args.fahrenheit]], dtype=torch.float32)

    with torch.no_grad():
        scaled_prediction = model(fahrenheit_tensor / fahrenheit_scale)
        celsius = scaled_prediction.item() * celsius_scale

    print(f"{args.fahrenheit:.2f}°F is about {celsius:.2f}°C")


if __name__ == "__main__":
    main()
