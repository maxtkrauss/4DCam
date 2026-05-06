from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(r"Z:\4DCam\Software\4DCam\Classification\Mobile_Net\fold0_task_charts")


def main():
    modality_rows = [
        {
            "label": "106-channel polarization averaged reconstruction",
            "textiles": 87.88,
            "camouflage": 83.33,
        },
        {
            "label": "4-channel spectrum averaged reconstruction",
            "textiles": 93.94,
            "camouflage": 83.33,
        },
        {
            "label": "424-channel spectral polarimetric reconstruction",
            "textiles": 96.97,
            "camouflage": 86.67,
        },
        {
            "label": "1-channel polarization averaged Scatterogram",
            "textiles": 100.00,
            "camouflage": 90.00,
        },
        {
            "label": "4-channel Scatterogram",
            "textiles": 100.00,
            "camouflage": 93.33,
        },
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_specs = [
        {
            "name": "Textiles",
            "slug": "textiles",
            "color": "#4C78A8",
            "values": [row["textiles"] for row in modality_rows],
        },
        {
            "name": "Camouflage",
            "slug": "camouflage",
            "color": "#F58518",
            "values": [row["camouflage"] for row in modality_rows],
        },
    ]

    labels = [row["label"] for row in modality_rows]

    for spec in task_specs:
        values = np.array(spec["values"])

        fig, ax = plt.subplots(figsize=(10, 4.8))
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=spec["color"], height=0.58)

        ax.set_title(f"Fold 0 {spec['name']} Modality Comparison")
        ax.set_xlabel("Best Test Accuracy (%)")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 110)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.invert_yaxis()

        for bar, value in zip(bars, values):
            ax.text(
                value + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center",
                ha="left",
                fontsize=10,
            )

        fig.tight_layout()
        png_path = OUTPUT_DIR / f"{spec['slug']}_fold0_modality_comparison.png"
        svg_path = OUTPUT_DIR / f"{spec['slug']}_fold0_modality_comparison.svg"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot to {png_path}")
        print(f"Wrote plot to {svg_path}")


if __name__ == "__main__":
    main()
