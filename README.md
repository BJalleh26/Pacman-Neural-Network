# Pacman-Classifier

## Overview

This project uses a MultiLayer Perceptron Neural Network classifier to decide what moves Pacman should make in the classic Pacman game.

The classifier is trained on 127 moves in good-moves.txt. These moves are represented as feature vectors of 1s and 0s (look in api.py for the structure of the feature vector).

The self built Neural Network is contained in classifier.py.

## Installation

To run the Pacman Classifier project, ensure you have Python 3.10 installed. Additionally, you need to install NumPy:

```bash
pip install numpy
```

## Execution
```bash
python pacman.py -p ClassifierAgent
```
