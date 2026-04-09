import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_med(y_true, y_pred):
    """
    y_true: [(x, y), ...]
    y_pred: [(x, y), ...]
    """
    distances = []

    for (x_t, y_t), (x_p, y_p) in zip(y_true, y_pred):
        d = euclidean_distance(x_t, y_t, x_p, y_p)
        distances.append(d)

    return np.mean(distances)


def compute_med_by_source(data, preds):
    """
    data: исходный dataset (с source)
    preds: предсказания [(x, y), ...]

    Возвращает:
    - MED для top
    - MED для bottom
    """
    top_true, top_pred = [], []
    bottom_true, bottom_pred = [], []

    for item, pred in zip(data, preds):
        true_point = (item["x_dst"], item["y_dst"])

        if item["source"] == "top":
            top_true.append(true_point)
            top_pred.append(pred)
        else:
            bottom_true.append(true_point)
            bottom_pred.append(pred)

    med_top = compute_med(top_true, top_pred)
    med_bottom = compute_med(bottom_true, bottom_pred)

    return med_top, med_bottom