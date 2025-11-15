import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_experiments(json_path: Path) -> List[Dict]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    experiments = payload.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in the provided JSON file.")
    return experiments


def plot_revenue_vs_k(experiments: List[Dict], output_path: Path) -> None:
    experiments = sorted(experiments, key=lambda item: item.get("k", 0))
    k_values = [exp.get("k") for exp in experiments]
    revenues = [exp.get("revenue") for exp in experiments]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(
        k_values,
        revenues,
        marker="o",
        linewidth=2.5,
        markersize=6,
        color="#3366cc",
    )

    for k, revenue in zip(k_values, revenues):
        ax.annotate(
            f"{revenue:,.0f}",
            xy=(k, revenue),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_title("Revenue Variation Across Congestion Factor k", fontsize=14, pad=12)
    ax.set_xlabel("Congestion factor k", fontsize=12)
    ax.set_ylabel("Total revenue", fontsize=12)
    ax.set_xticks(k_values)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot revenue variation across different values of k."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("revenue_variation_with_k.json"),
        help="Path to the JSON file containing experiment data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/revenue_vs_k.png"),
        help="Path for the exported plot image.",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.input)
    plot_revenue_vs_k(experiments, args.output)
    print(f"Plot saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
