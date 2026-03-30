def normalize_scalar(field):
    return (field - field.min()) / (field.max() - field.min() + 1e-8)