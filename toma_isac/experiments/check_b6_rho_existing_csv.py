from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "results"
    / "stage8"
    / "stage8_sweep_rho_si.csv"
)


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    table = df.pivot(
        index="scene_seed",
        columns="parameter_value",
        values="b6_wsr",
    )

    # 按 rho 从小到大排列。
    table = table.reindex(
        columns=[
            1.0e-3,
            1.0e-2,
            1.0e-1,
            1.0,
        ]
    )

    print("\nB6 = MA + FP, scene-level rho_si results\n")
    print(table.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nMonotonicity check\n")

    abnormal_scenes = []

    for scene_seed, row in table.iterrows():
        values = row.to_numpy(dtype=float)

        differences = values[1:] - values[:-1]

        abnormal = differences > 1.0e-8

        status = (
            "ABNORMAL"
            if abnormal.any()
            else "OK"
        )

        print(
            f"scene {scene_seed}: "
            f"{status} | "
            f"steps={differences}"
        )

        if abnormal.any():
            abnormal_scenes.append(
                int(scene_seed)
            )

    print("\nAverage over scenes:")
    print(
        table.mean(axis=0).to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print(
        "\nAbnormal scenes:",
        abnormal_scenes,
    )

    print(
        f"Number abnormal = "
        f"{len(abnormal_scenes)}/{len(table)}"
    )


if __name__ == "__main__":
    main()