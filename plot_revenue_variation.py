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


def build_label(order_id: int, steps: List[Dict]) -> str:
    sequence = []
    for entry in steps:
        route = entry.get("added_route", {})
        src = route.get("src", "?")
        dst = route.get("dst", "?")
        sequence.append(f"{src}->{dst}")
    routes_str = " | ".join(sequence)
    return f"Order {order_id}: {routes_str}"


def plot_revenue_curves(experiments: List[Dict], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for idx, experiment in enumerate(experiments):
        order_id = experiment.get("order_id")
        steps = experiment.get("steps", [])
        if not steps:
            continue

        x_values = [step.get("step") for step in steps]
        y_values = [step.get("revenue") for step in steps]
        label = build_label(order_id, steps)

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=label,
        )

        final_x, final_y = x_values[-1], y_values[-1]
        dx, dy = [(10, 8), (10, -14), (-35, 8), (-35, -14)][idx % 4]
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.annotate(
            f"{final_y:,.0f}",
            xy=(final_x, final_y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "alpha": 0.7,
                "ec": "none",
            },
        )

    ax.set_title(
        "Revenue Impact of Adding Routes in Different Orders", fontsize=14, pad=12
    )
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cumulative Revenue", fontsize=12)
    ax.set_xticks(
        sorted(
            {
                step
                for exp in experiments
                for step in [s.get("step") for s in exp.get("steps", [])]
            }
        )
    )
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot revenue variation across route addition orders."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("revenue_variation_with_adding_routes.json"),
        help="Path to the JSON file containing experiment data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/revenue_variation.png"),
        help="Path for the exported plot image.",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.input)
    plot_revenue_curves(experiments, args.output)
    print(f"Plot saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
