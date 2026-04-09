import json
import os
from tqdm import tqdm


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_split(data_root):
    path = os.path.join(data_root, "split.json")
    return load_json(path)


def parse_points(points_list):
    """
    Преобразует список точек в dict:
    number -> (x, y)
    """
    result = {}
    for p in points_list:
        number = p["number"]
        x = p["x"]
        y = p["y"]
        result[number] = (x, y)
    return result


def process_pair(pair, source_type):
    """
    Обработка одну пару кадров
    """
    door2_points = parse_points(pair["image1_coordinates"])
    src_points = parse_points(pair["image2_coordinates"])

    data = []

    

    # проходим по общим точкам
    for number in door2_points:
        if number not in src_points:
            continue

        x_dst, y_dst = door2_points[number]   # door2
        x_src, y_src = src_points[number]     # top или bottom

        data.append({
            "x_src": x_src,
            "y_src": y_src,
            "x_dst": x_dst,
            "y_dst": y_dst,
            "source": source_type
        })

    return data


def load_session(session_path):
    """
    Загрузка сессии
    """
    session_data = []

    #  TOP 
    top_path = os.path.join(session_path, "coords_top.json")
    if os.path.exists(top_path):
        top_json = load_json(top_path)

        for pair in top_json:
            session_data.extend(process_pair(pair, "top"))

    #  BOTTOM 
    bottom_path = os.path.join(session_path, "coords_bottom.json")
    if os.path.exists(bottom_path):
        bottom_json = load_json(bottom_path)

        for pair in bottom_json:
            session_data.extend(process_pair(pair, "bottom"))

    return session_data


def build_dataset(data_root, split_type="train"):
    split = load_split(data_root)

    data = []

    for session_rel_path in tqdm(split[split_type]):
        session_path = os.path.join(data_root, session_rel_path)

        session_data = load_session(session_path)
        data.extend(session_data)

        return data
    
def process_pair_to_lists(pair, source_type):
    door2_points = parse_points(pair["image1_coordinates"])
    src_points_dict = parse_points(pair["image2_coordinates"])

    common_numbers = sorted(set(door2_points.keys()) & set(src_points_dict.keys()))

    src_points = []
    dst_points = []

    for number in common_numbers:
        x_dst, y_dst = door2_points[number]
        x_src, y_src = src_points_dict[number]

        src_points.append((x_src, y_src))
        dst_points.append((x_dst, y_dst))

    if len(src_points) < 4:
        return None

    return {
        "src_points": src_points,
        "dst_points": dst_points,
        "source": source_type
    }

def load_session_pairs(session_path):
    session_data = []

    #  TOP 
    top_path = os.path.join(session_path, "coords_top.json")
    if os.path.exists(top_path):
        top_json = load_json(top_path)

        for pair in top_json:
            item = process_pair_to_lists(pair, "top")
            if item is not None:
                session_data.append(item)

    #  BOTTOM 
    bottom_path = os.path.join(session_path, "coords_bottom.json")
    if os.path.exists(bottom_path):
        bottom_json = load_json(bottom_path)

        for pair in bottom_json:
            item = process_pair_to_lists(pair, "bottom")
            if item is not None:
                session_data.append(item)

    return session_data

def build_pairs_dataset(data_root, split_type="train"):
    split = load_split(data_root)

    data = []

    for session_rel_path in split[split_type]:
        session_path = os.path.join(data_root, session_rel_path)

        session_data = load_session_pairs(session_path)
        data.extend(session_data)

    return data
