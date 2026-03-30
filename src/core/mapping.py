def scalar_to_target(field, contrast="high"):
    if contrast == "low":
        return 0.35 + field * (0.65 - 0.35)
    elif contrast == "high":
        return 0.2 + field * (0.8 - 0.2)
    else:
        raise ValueError("Unknown contrast")