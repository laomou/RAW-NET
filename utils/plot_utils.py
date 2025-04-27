from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_logs(logs, fields=('lr', 'loss'), ewm_col=0, log_name='log.txt'):
    func_name = "plot_utils.py::plot_logs"
    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n Expect list[Path] or single Path obj, received {type(logs)}")

    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            df.interpolate().ewm(com=ewm_col).mean().plot(
                y=[f'train_{field}'],
                ax=axs[j],
                color=[color] * 2,
                style=['-', '--']
            )

    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)

    plt.tight_layout()

    plt.savefig("plot.png", format="png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', type=Path)
    args = parser.parse_args()

    plot_logs(args.logs)
