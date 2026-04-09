import numpy as np
import pickle

WIDTH = 3200
HEIGHT = 1800


def prepare_single(x, y, source):
    cx = WIDTH / 2
    cy = HEIGHT / 2

    x_norm = x / WIDTH
    y_norm = y / HEIGHT

    dx = (x - cx) / WIDTH
    dy = (y - cy) / HEIGHT

    source_val = 0 if source == "top" else 1

    return np.array([[x_norm, y_norm, dx, dy, source_val]])


class Predictor:
    def __init__(self, model_path="model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, x, y, source):
        X = prepare_single(x, y, source)
        pred = self.model.predict(X)[0]
        return float(pred[0]), float(pred[1])