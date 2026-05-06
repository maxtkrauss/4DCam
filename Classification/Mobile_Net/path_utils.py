from pathlib import Path, PureWindowsPath


WINDOWS_DATASET_ROOT = PureWindowsPath(r"D:\4DCam_Data\Textile-Camo Classification")


def remap_dataset_path(path_str: str, dataset_root: str | None = None) -> str:
    if not dataset_root:
        return path_str

    try:
        windows_path = PureWindowsPath(path_str)
    except Exception:
        return path_str

    try:
        relative = windows_path.relative_to(WINDOWS_DATASET_ROOT)
    except Exception:
        return path_str

    return str(Path(dataset_root) / Path(*relative.parts))
