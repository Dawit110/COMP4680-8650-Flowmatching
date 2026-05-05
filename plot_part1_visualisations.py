from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataloader import ToyDiffusionDataset, AVAILABLE_DATASETS, AVAILABLE_DIMS

# Creating the figures directory that holds the visiulized data
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Loading in a dataset of dimension D
def load_points_2d(dataset_name: str, D: int) -> np.ndarray:
    """
    Load a dataset at dimension D.

    If D=2, the data is already 2D.
    If D=8 or D=32, project it back to 2D using the provided to_2d method.
    """
    dataset = ToyDiffusionDataset(name=dataset_name, dim=D)

    samples = dataset.data.numpy()
    points_2d = dataset.to_2d(samples)
    return points_2d

def plot_individual_figures():
    for dataset_name in AVAILABLE_DATASETS:
        for dim in AVAILABLE_DIMS:
            points = load_points_2d(dataset_name, dim)

            plt.figure(figsize=(5, 5))
            plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.5)
            plt.title(f"{dataset_name}, D={dim}, projected to 2D")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            output_path = FIG_DIR / f"part1_{dataset_name}_{dim}d.png"
            plt.savefig(output_path, dpi=200)
            plt.close()

            print(f"Saved {output_path}")


def plot_grid_figure():
    fig, axes = plt.subplots(
        nrows=len(AVAILABLE_DATASETS),
        ncols=len(AVAILABLE_DIMS),
        figsize=(12, 10),
    )

    for row, dataset_name in enumerate(AVAILABLE_DATASETS):
        for col, dim in enumerate(AVAILABLE_DIMS):
            points = load_points_2d(dataset_name, dim)

            ax = axes[row, col]
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.5)
            ax.set_title(f"{dataset_name}, D={dim}")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Toy datasets projected to 2D", fontsize=16)
    fig.tight_layout()

    output_path = FIG_DIR / "part1_dataset_visualisation_grid.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_individual_figures()
    plot_grid_figure()