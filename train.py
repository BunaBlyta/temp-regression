from pathlib import Path

import torch


MODEL_PATH = Path("temperature_model.pt")

# Scaling keeps the numbers small, which helps SGD train smoothly.
FAHRENHEIT_SCALE = 100.0
CELSIUS_SCALE = 100.0


def main():
    torch.manual_seed(42)

    fahrenheit = torch.tensor(
        [[-40.0], [0.0], [32.0], [50.0], [68.0], [86.0], [98.6], [104.0], [122.0], [212.0]]
    )
    celsius = (fahrenheit - 32.0) * 5.0 / 9.0

    x_train = fahrenheit / FAHRENHEIT_SCALE
    y_train = celsius / CELSIUS_SCALE

    model = torch.nn.Linear(1, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epochs = 5000

    for epoch in range(1, epochs + 1):
        predictions = model(x_train)
        loss = loss_fn(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.8f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fahrenheit_scale": FAHRENHEIT_SCALE,
            "celsius_scale": CELSIUS_SCALE,
        },
        MODEL_PATH,
    )

    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
