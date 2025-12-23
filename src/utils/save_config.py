import csv
import os


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_config_csv(cfg, save_dir, filename="config.csv"):
    if hasattr(cfg, "config"):
        cfg_dict = cfg.config
    else:
        cfg_dict = cfg.__dict__

    flat_cfg = flatten_dict(cfg_dict)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in sorted(flat_cfg.items()):
            writer.writerow([k, v])

    print(f"Config saved to: {csv_path}")
