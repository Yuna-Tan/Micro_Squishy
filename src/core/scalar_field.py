# def normalize_scalar(field):
#     return (field - field.min()) / (field.max() - field.min() + 1e-8)
import numpy as np

def normalize_scalar(field):
    min_v = field.min()
    max_v = field.max()

    if abs(max_v - min_v) < 1e-8:
        print("⚠️ Uniform field detected → skip normalization")
        return np.ones_like(field)  # ⭐关键

    return (field - min_v) / (max_v - min_v)