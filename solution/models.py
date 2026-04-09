from sklearn.ensemble import RandomForestRegressor
import numpy as np


class CoordinateModel:
    def __init__(self):
        self.model_top_x = RandomForestRegressor(n_estimators=150, max_depth=10)
        self.model_top_y = RandomForestRegressor(n_estimators=150, max_depth=10)

        self.model_bottom_x = RandomForestRegressor(n_estimators=150, max_depth=10)
        self.model_bottom_y = RandomForestRegressor(n_estimators=150, max_depth=10)

    def fit(self, X, y, sources):
        top_idx = sources == 0
        bottom_idx = sources == 1

        self.model_top_x.fit(X[top_idx], y[top_idx, 0])
        self.model_top_y.fit(X[top_idx], y[top_idx, 1])

        self.model_bottom_x.fit(X[bottom_idx], y[bottom_idx, 0])
        self.model_bottom_y.fit(X[bottom_idx], y[bottom_idx, 1])

    def predict(self, X, sources):
        preds = []

        for i in range(len(X)):
            if sources[i] == 0:
                x = self.model_top_x.predict(X[i:i+1])[0]
                y = self.model_top_y.predict(X[i:i+1])[0]
            else:
                x = self.model_bottom_x.predict(X[i:i+1])[0]
                y = self.model_bottom_y.predict(X[i:i+1])[0]

            preds.append([x, y])

        return np.array(preds)