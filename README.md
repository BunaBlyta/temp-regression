# PyTorch Temperature Regression

This is a small beginner-friendly PyTorch project that trains a simple regression model to convert Fahrenheit temperatures to Celsius.

The model learns from examples like:

```text
32°F  -> 0°C
212°F -> 100°C
```

Instead of hard-coding the formula, the project uses a neural network with one input and one output:

```python
torch.nn.Linear(1, 1)
```

That means the model learns a straight-line relationship between Fahrenheit and Celsius.

## Project Structure

```text
pytorch-temp-regression/
├── README.md
├── requirements.txt
├── train.py
├── predict.py
└── .gitignore
```

## Key Ideas

### Tensors

A tensor is PyTorch's main data container. In this project, the Fahrenheit and Celsius values are stored as tensors so PyTorch can train with them.

### Model

The model is a small neural network. Here it is only one layer:

```python
torch.nn.Linear(1, 1)
```

It takes one Fahrenheit value and predicts one Celsius value.

### Loss

Loss measures how wrong the model is. This project uses mean squared error:

```python
torch.nn.MSELoss()
```

If the prediction is far from the correct Celsius value, the loss is larger.

### Backpropagation

Backpropagation is how PyTorch figures out how each model weight contributed to the error. Calling `loss.backward()` calculates the gradients needed to improve the model.

### Optimizer

The optimizer updates the model's weights. This project uses stochastic gradient descent:

```python
torch.optim.SGD
```

After backpropagation, `optimizer.step()` changes the model a little so the next prediction should be better.

### Epochs

An epoch is one full training pass over the dataset. Training for many epochs gives the model many chances to improve.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train.py
```

This trains the model and saves it to:

```text
temperature_model.pt
```

## Make a Prediction

After training, run:

```bash
python predict.py 98.6
```

Example output:

```text
98.60°F is about 37.00°C
```

## What I Learned

- How to create tensors for numeric training data.
- How to build a simple `torch.nn.Linear` regression model.
- How loss measures the difference between predictions and correct answers.
- How backpropagation and an optimizer help a model improve.
- How to save a trained PyTorch model and load it later for prediction.
