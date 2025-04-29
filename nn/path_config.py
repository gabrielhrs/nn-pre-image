import os
import platform
import getpass

from typing import Any, Dict


def get_paths() -> Dict[str, Any]:
    if 'Darwin' == platform.system():
        tilde = os.path.expanduser("~")

        project_base = os.path.join(tilde, "")
    else:
        project_base = os.getcwd()

    model_state_dict_folder = os.path.join(project_base, "model_states")
    cached_dataset_folder = os.path.join(project_base, "data")

    writeup_dir = os.path.join(project_base, "writeup")
    plot_dir = os.path.join(writeup_dir, "plots")
    results_folder = os.path.join(project_base, "results")
    cached_calculations = os.path.join(project_base, "cached_calculations")
    lunar_lander_checkpoints = os.path.join(project_base, "lunar_lander_checkpoints")
    acas_dir = os.path.join(project_base, "acas")

    # just a general purpose stashing area, that can be changed at basically arbitrary places
    work_dir = os.path.join(project_base, "work")

    paths = {
        "acas": acas_dir,
        "project_base": project_base,
        "cached_calculations": cached_calculations,
        "lunar_lander_checkpoints": lunar_lander_checkpoints,
        "model_state_dict": model_state_dict_folder,
        "cached_datasets": cached_dataset_folder,
        "plots": plot_dir,
        "results": results_folder,
        "work": work_dir
    }

    for k, v in paths.items():
        os.makedirs(v, exist_ok=True)
    return paths


if __name__ == "__main__":
    paths = get_paths()