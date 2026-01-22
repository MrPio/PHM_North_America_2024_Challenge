import os
from ipywidgets import widgets
from IPython.display import display
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


def __plot(
    losses: tuple[list, list, list],
    train=False,
    valid=False,
    title=None,
    scale=1,
    ax=None,
    standalone=True,
    label=None,
    metric="test_loss",
    ylim=None,
    tasks: list[int] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(20 * scale, 10 * scale), dpi=150)
        ax.set_xticks(range(0, len(losses[0]) * len(losses[0][0]) + 1, len(losses[0][0])))
        for i, test_loss in enumerate(losses[0]):
            ax.add_line(
                Line2D([(i + 1) * len(test_loss)] * 2, [0, 99], linestyle="--", color="black", linewidth=1, alpha=0.5)
            )
            ax.fill_between(
                range(i * len(test_loss), (i + 1) * len(test_loss) + 1),
                -99,
                99,
                alpha=0.25,
                color="tab:orange" if i % 2 == 0 else "tab:blue",
                label=None if i > 1 else "np > ng" if i % 2 == 0 else "np < ng",
            )
        ax.set_ylim(ylim if ylim else [-1, 0] if min(losses[0][-1]) < 0 else [0, 2])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(" ".join(map(lambda s: s.capitalize(), metric.split("_"))))
    if train:
        ax.plot([x for xs in losses[0] for x in xs], color="tab:orange", label="train")
    if valid:
        ax.plot([x for xs in losses[1] for x in xs], color="tab:green", label="valid")

    ax.plot(
        [
            (
                sum(x[t][metric] * x[t]["samples"] for t in tasks) / sum(x[t]["samples"] for t in tasks)
                if tasks
                else x[metric]
            )
            for xs in losses[2]
            for x in xs
        ],
        color="tab:blue" if standalone else None,
        label="test" if label is None else label,
        linewidth=5,
    )

    if title:
        ax.set_title(title)
    if standalone:
        ax.legend()
        plt.show()
    return ax


def continual_learning_gui(results_dir: Path, metric="test_loss", ylim=None, title=None, loc="best", tasks=None):
    output = widgets.Output()
    selected_files = []
    buttons = {}
    losses = {}
    last_ax = {"ax": None}

    def select_file(file, deselect_all=False):
        with output:
            try:
                output.clear_output()
                if deselect_all:
                    selected_files.clear()
                else:
                    if file in selected_files:
                        selected_files.remove(file)
                    else:
                        selected_files.append(file)

                # Set button styles
                for f, button in buttons.items():
                    button.button_style = "primary" if f in selected_files else ""

                # Plot
                ax = None

                def format_label(file):
                    pieces = file.replace("_layers=", "").replace("EfficientKAN", "KAN").split("_")[:-1]
                    pieces = filter(
                        lambda piece: not any(x in piece for x in ["epoch", "times"]) and len(piece) > 1, pieces
                    )
                    return ", ".join(pieces)

                for selected_file in selected_files:
                    ax = __plot(
                        losses[selected_file],
                        title=title if title else file,
                        train=False,
                        ax=ax,
                        standalone=False,
                        label=format_label(selected_file),
                        metric=metric,
                        ylim=ylim,
                        tasks=tasks,
                    )
                if ax:
                    ax.legend(loc=loc)
                    last_ax["ax"] = ax
                    plt.show()
                save.layout.display = "block" if ax else "none"
            except Exception as e:
                print(e)

    def save_fig():
        if last_ax["ax"] is not None:
            fig = last_ax["ax"].get_figure()
            filename = f"{last_ax['ax'].get_title() or 'plot'}.png"
            fig.savefig(f"img/{filename}", dpi=300, bbox_inches="tight")

    files = results_dir.glob("*")
    files = sorted(files, key=lambda f: os.path.getctime(results_dir / f), reverse=False)

    for file in filter(lambda f: not f.is_dir(), files):
        with file.open("rb") as f:
            losses[file.name] = pickle.load(f)

        btn = widgets.Button(
            description=" ——— ".join(file.name.split("_")[:-1]),
            layout=widgets.Layout(width="auto"),
            style={"font_size": "20px", "text_decoration": "underline" if min(losses[file.name][0][-1]) < 0 else ""},
            button_style="",
        )
        btn.on_click(lambda _, f=file.name: select_file(f))
        buttons[file.name] = btn
    clear = widgets.Button(
        description="✖️ Clear",
        layout=widgets.Layout(width="auto"),
        style={"font_size": "20px", "font_weight": ""},
        button_style="warning",
    )
    clear.on_click(lambda _: select_file(None, deselect_all=True))
    save = widgets.Button(
        description="💾 Save Plot",
        layout=widgets.Layout(width="auto"),
        style={"font_size": "20px", "font_weight": ""},
        button_style="success",
    )
    save.layout.display = "none"
    save.on_click(lambda _: save_fig())
    spacer = widgets.HTML(value="<div style='margin-top:30px;'></div>")
    display(widgets.VBox([*buttons.values(), clear, spacer, output, save]))
