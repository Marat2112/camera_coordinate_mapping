import numpy as np
import pickle


# признаки, что и в train
def create_features(x, y, source, img_w=3840, img_h=2160):
    x_norm = x / img_w
    y_norm = y / img_h

    dx = x_norm - 0.5
    dy = y_norm - 0.5

    r2 = dx**2 + dy**2

    source_flag = 1 if source == "top" else 0

    return [
        x_norm,
        y_norm,
        dx,
        dy,
        x_norm * y_norm,
        r2,
        source_flag   
    ]

class Predictor:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, x, y, source):
        # 1. признак
        features = create_features(x, y, source)

        # 2. nump
        X = np.array([features])

        # 3. ПРЕДСКАЗ
        pred = self.model.predict(X, [source])[0]

        return float(pred[0]), float(pred[1])



if __name__ == "__main__":
    predictor = Predictor("model.pkl")

    print("=== Coordinate Predictor ===")

    # ввод 
    try:
        x = float(input("Enter x: "))
        y = float(input("Enter y: "))
        source = input("Enter source (top/bottom): ").strip().lower()

        if source not in ["top", "bottom"]:
            raise ValueError("source must be 'top' or 'bottom'")

        x_pred, y_pred = predictor.predict(x, y, source)

        print("\nResult:")
        print("Input :", (x, y, source))
        print("Output:", (x_pred, y_pred))

    except Exception as e:
        print("Error:", e)