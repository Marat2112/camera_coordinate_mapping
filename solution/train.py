import numpy as np
import pickle

from dataset import build_dataset
from models import CoordinateModel
from metrics import compute_med_by_source



def prepare_ml_data(data):
    X = []
    y = []

    WIDTH = 3200
    HEIGHT = 1800

    cx = WIDTH / 2
    cy = HEIGHT / 2

    for d in data:
        x = d["x_src"]
        y_src = d["y_src"]

        # нормализация
        x_norm = x / WIDTH
        y_norm = y_src / HEIGHT

        dx = (x - cx) / WIDTH
        dy = (y_src - cy) / HEIGHT

        # дополнительные признаки
        r2 = dx**2 + dy**2
        xy = x_norm * y_norm

        source = 0 if d["source"] == "top" else 1

        X.append([
            x_norm,
            y_norm,
            dx,
            dy,
            r2,
            xy,
            source
        ])

        y.append([d["x_dst"], d["y_dst"]])

    return np.array(X), np.array(y)



# 2. (top/bottom)

def extract_sources(data):
    return np.array([0 if d["source"] == "top" else 1 for d in data])



# 3. MAIN


# Загружаем 
train_data = build_dataset("../data", "train")
val_data = build_dataset("../data", "val")

print("Train size:", len(train_data))
print("Val size:", len(val_data))



X_train, y_train = prepare_ml_data(train_data)
X_val, y_val = prepare_ml_data(val_data)

# Источник
sources_train = extract_sources(train_data)
sources_val = extract_sources(val_data)



# 4. Обучение 

model = CoordinateModel()
model.fit(X_train, y_train, sources_train)



# 5. Предсказание

preds = model.predict(X_val, sources_val)

preds_list = [(p[0], p[1]) for p in preds]



# 6. Метрика

med_top, med_bottom = compute_med_by_source(val_data, preds_list)

print("ML MED top:", med_top)
print("ML MED bottom:", med_bottom)



# 7. Сохранение

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")
