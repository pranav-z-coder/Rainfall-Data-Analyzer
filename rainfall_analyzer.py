"""
PS No. 26: Rainfall Data Analyzer
Rungta International Skills University - School of Computer Science and Engineering
B.Tech Computer Science and Engineering

Concepts: Python 3, NumPy, Pandas, Matplotlib, Statistics, Boolean Filtering
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def format_num(val):
    """Format whole numbers as integers (e.g., 700) and decimals to 2 places."""
    if float(val).is_integer():
        return str(int(round(val)))
    return f"{val:.2f}"


def load_rainfall_csv(filepath="rain.csv"):
    """
    Dynamically load monthly rainfall data from an existing CSV file using Pandas
    and validate missing, non-numeric, or negative readings.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found. Please place '{filepath}' in the project folder.")
        return pd.DataFrame(columns=["Month", "Rainfall_mm"])

    try:
        raw_df = pd.read_csv(filepath, header=None, dtype=str, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        print("CSV file is empty.")
        return pd.DataFrame(columns=["Month", "Rainfall_mm"])
    except pd.errors.ParserError as exc:
        print(f"Error: Could not parse CSV file: {exc}")
        return pd.DataFrame(columns=["Month", "Rainfall_mm"])

    if raw_df.empty:
        print("CSV file is empty.")
        return pd.DataFrame(columns=["Month", "Rainfall_mm"])

    # Automatically skip header row if present (e.g., Month, Rainfall_mm)
    first_cell = str(raw_df.iloc[0, 0]).strip().lower()
    if first_cell in ("month", "months"):
        raw_df = raw_df.iloc[1:].reset_index(drop=True)

    valid_rows = []
    had_errors = False

    for idx, row in raw_df.iterrows():
        row_num = idx + 1
        month = str(row[0]).strip() if pd.notna(row[0]) else ""
        raw_rain = str(row[1]).strip() if len(row) > 1 and pd.notna(row[1]) else ""

        if not month or raw_rain == "" or raw_rain.lower() == "nan":
            print(f"Row {row_num} skipped: invalid Rainfall '{raw_rain}'.")
            had_errors = True
            continue

        try:
            rain_val = float(raw_rain)
        except ValueError:
            print(f"Row {row_num} skipped: invalid Rainfall '{raw_rain}'.")
            had_errors = True
            continue

        if not np.isfinite(rain_val):
            print(f"Row {row_num} skipped: Rainfall must be a finite number.")
            had_errors = True
            continue

        if rain_val < 0:
            print(f"Row {row_num} skipped: Rainfall cannot be negative.")
            had_errors = True
            continue

        valid_rows.append({"Month": month, "Rainfall_mm": rain_val})

    valid_df = pd.DataFrame(valid_rows, columns=["Month", "Rainfall_mm"])

    if had_errors:
        total_valid = np.sum(valid_df["Rainfall_mm"].to_numpy(dtype=float)) if not valid_df.empty else 0
        print(f"Valid rows: {len(valid_df)} | Total: {format_num(total_valid)} mm")

    return valid_df


def compute_metrics(df):
    """Calculate all statistical metrics dynamically using NumPy."""
    months = df["Month"].to_numpy()
    rainfall = df["Rainfall_mm"].to_numpy(dtype=float)

    total_rain = np.sum(rainfall)
    mean_rain = np.mean(rainfall)
    median_rain = np.median(rainfall)
    std_dev = np.std(rainfall)  # Population standard deviation (np.std default)
    max_idx = int(np.argmax(rainfall))
    min_idx = int(np.argmin(rainfall))

    return {
        "months": months,
        "rainfall": rainfall,
        "total": total_rain,
        "mean": mean_rain,
        "median": median_rain,
        "std": std_dev,
        "max_idx": max_idx,
        "min_idx": min_idx,
        "max_val": rainfall[max_idx],
        "min_val": rainfall[min_idx],
        "max_month": months[max_idx],
        "min_month": months[min_idx],
    }


def build_stats_box_text(m):
    """Return a formatted text summary of all metrics to embed inside any chart."""
    return (
        f"Total: {format_num(m['total'])} mm\n"
        f"Mean (Avg): {m['mean']:.2f} mm\n"
        f"Median: {m['median']:.2f} mm\n"
        f"Std Dev: {m['std']:.2f} mm\n"
        f"Max: {m['max_month']} ({format_num(m['max_val'])} mm)\n"
        f"Min: {m['min_month']} ({format_num(m['min_val'])} mm)"
    )


def analyze_statistics(df):
    """
    Menu Option 1: Calculate Total, Mean, Maximum, Minimum, Population Std Dev (np.std),
    and detect heavy-rainfall months (> 200 mm) using NumPy boolean filtering.
    """
    if df.empty:
        print("No valid rainfall data available for analysis.")
        return

    m = compute_metrics(df)
    rainfall = m["rainfall"]
    months = m["months"]

    # Boolean filtering for heavy-rainfall months (> 200 mm)
    heavy_mask = rainfall > 200
    heavy_months = months[heavy_mask]

    if len(rainfall) == 1:
        print(
            f"Total: {format_num(m['total'])} mm | "
            f"Mean: {m['mean']:.2f} | "
            f"Std Dev: {m['std']:.2f}"
        )
    else:
        print(f"Total: {format_num(m['total'])} mm | Mean: {m['mean']:.2f}")
        print(
            f"Max: {m['max_month']} ({format_num(m['max_val'])}) | "
            f"Min: {m['min_month']} ({format_num(m['min_val'])})"
        )
        print(f"Std Dev: {m['std']:.2f}")

    if len(heavy_months) > 0:
        print(f"Heavy rainfall months: {', '.join(heavy_months)}")
    else:
        print("No month above 200 mm")


def plot_rainfall_trend(df):
    """
    Menu Option 2:
    Displays rainfall/cumulative trends and a statistical summary.
    Layout and annotation density adapt to the number of months to prevent
    labels from colliding or being clipped at the chart edges.
    """
    if df.empty:
        print("No valid data available to plot.")
        return

    m = compute_metrics(df)
    months = m["months"]
    rainfall = m["rainfall"]
    cumulative = np.cumsum(rainfall)
    count = len(months)

    print(
        f"Trend Summary -> Total: {format_num(m['total'])} mm | "
        f"Mean: {m['mean']:.2f} mm | Median: {m['median']:.2f} mm | Std Dev: {m['std']:.2f} mm"
    )
    print("Trend and statistical charts displayed")

    # Wider figures for many months; labels are rotated instead of being squeezed.
    fig_width = max(13, min(20, 13 + max(0, count - 12) * 0.22))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 6.2))
    fig.suptitle(
        "Option 2: Monthly Rainfall Trend, Cumulative Total & Statistical Summary",
        fontsize=13,
        fontweight="bold",
    )

    x_pos = np.arange(count)
    ax1.plot(
        x_pos, rainfall, marker="o", color="#ff7f0e",
        linewidth=2.2, markersize=6 if count <= 16 else 4,
        label="Monthly Rainfall (mm)",
    )
    ax1.plot(
        x_pos, cumulative, marker="s", color="#1f77b4",
        linestyle=":", linewidth=1.8, markersize=5 if count <= 16 else 3.5,
        label=f"Cumulative Total (Final: {format_num(m['total'])} mm)",
    )
    ax1.axhline(m["mean"], color="green", linestyle="--", linewidth=1.5,
                label=f"Mean/Avg ({m['mean']:.2f} mm)")
    ax1.axhline(m["median"], color="purple", linestyle="-.", linewidth=1.3,
                label=f"Median ({m['median']:.2f} mm)")
    ax1.fill_between(
        x_pos, m["mean"] - m["std"], m["mean"] + m["std"],
        color="green", alpha=0.13,
        label=f"±1 Std Dev ({m['std']:.2f} mm)",
    )
    ax1.scatter(
        [m["max_idx"]], [m["max_val"]], color="red", s=65,
        zorder=5, label=f"Max: {m['max_month']} ({format_num(m['max_val'])})",
    )
    ax1.scatter(
        [m["min_idx"]], [m["min_val"]], color="blue", s=65,
        zorder=5, label=f"Min: {m['min_month']} ({format_num(m['min_val'])})",
    )

    # Dense per-point labels become unreadable on long datasets. For <=12 months,
    # show all values; for larger datasets, rely on the plotted points and legend
    # and label only the final cumulative value.
    if count <= 6:
        for i, (r_val, c_val) in enumerate(zip(rainfall, cumulative)):
            ax1.annotate(
                format_num(r_val), (x_pos[i], r_val),
                textcoords="offset points", xytext=(0, 7 if i % 2 == 0 else -12),
                ha="center", va="bottom" if i % 2 == 0 else "top",
                fontsize=7.5, clip_on=False,
            )
            if i > 0:
                ax1.annotate(
                    f"Σ={format_num(c_val)}", (x_pos[i], c_val),
                    textcoords="offset points", xytext=(0, 8 if i % 2 == 0 else -12),
                    ha="center", va="bottom" if i % 2 == 0 else "top",
                    fontsize=7, color="#1f77b4", clip_on=False,
                )
    elif count > 0:
        last = count - 1
        ax1.annotate(
            f"Final total: {format_num(cumulative[-1])} mm",
            (x_pos[-1], cumulative[-1]),
            textcoords="offset points", xytext=(-8, 10),
            ha="right", va="bottom", fontsize=8, fontweight="bold",
            color="#1f77b4", clip_on=False,
        )

    ax1.set_xticks(x_pos)
    tick_rotation = 90 if count > 16 else (45 if count > 8 else 0)
    ax1.set_xticklabels(
        months,
        rotation=tick_rotation,
        ha="right" if tick_rotation else "center",
        fontsize=7 if count > 16 else 8,
    )
    ax1.set_title("Monthly Rainfall Trend & Cumulative Total")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Rainfall (mm)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best", fontsize=7.5 if count > 12 else 8)
    ax1.margins(x=0.08, y=0.18)

    stat_labels = [
        "Total", "Mean (Avg)", "Median",
        f"Max ({m['max_month']})", f"Min ({m['min_month']})", "Std Dev",
    ]
    stat_values = [m["total"], m["mean"], m["median"], m["max_val"], m["min_val"], m["std"]]
    stat_colors = ["#4c78a8", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#f58518"]
    bars = ax2.bar(stat_labels, stat_values, color=stat_colors, edgecolor="black", alpha=0.85)
    ax2.set_title("All Statistical Metrics Overview (NumPy)")
    ax2.set_xlabel("Metric")
    ax2.set_ylabel("Rainfall (mm)")
    ax2.tick_params(axis="x", rotation=22, labelsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    max_stat = max([float(v) for v in stat_values] + [1.0])
    ax2.set_ylim(0, max_stat * 1.20)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            format_num(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            clip_on=False,
        )
    ax2.margins(x=0.08)

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.92])
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    plt.close(fig)

def compare_with_average_and_plot(df):
    """
    Menu Option 3:
    Compare each month's rainfall with the mean. For up to 16 months, use
    vertical bars with a dedicated in-plot summary panel. Above 16 months,
    switch to horizontal bars and a reserved top information strip so labels
    remain readable without making the figure excessively wide.
    """
    if df.empty:
        print("No valid rainfall data available.")
        return

    m = compute_metrics(df)
    months = m["months"]
    rainfall = m["rainfall"]
    mean_rain = m["mean"]
    deviations = rainfall - mean_rain
    count = len(months)
    above_mask = rainfall > mean_rain
    below_mask = rainfall < mean_rain

    above_items = [
        f"{mon} ({format_num(r)})" for mon, r in zip(months[above_mask], rainfall[above_mask])
    ]
    below_items = [
        f"{mon} ({format_num(r)})" for mon, r in zip(months[below_mask], rainfall[below_mask])
    ]
    print(f"Above average ({mean_rain:.2f}): {' | '.join(above_items) if above_items else 'None'}")
    print(f"Below average: {' | '.join(below_items) if below_items else 'None'}")
    print("Bar chart 'Monthly Rainfall' displayed")

    # Larger datasets (or long month names) use horizontal bars to keep labels readable.
    long_month_labels = max((len(str(month)) for month in months), default=0) > 8
    many_months = count > 12 or (count > 8 and long_month_labels)
    if many_months:
        fig_height = max(7.5, min(18.0, 3.0 + count * 0.25))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

    fig.suptitle(
        f"Option 3: Comparison with Average (Total: {format_num(m['total'])} mm | "
        f"Mean: {m['mean']:.2f} mm | Median: {m['median']:.2f} mm | "
        f"Std Dev: {m['std']:.2f} mm)",
        fontsize=11 if many_months else 12,
        fontweight="bold",
    )

    bar_colors = ["#d62728" if r > 200 else "#1f77b4" for r in rainfall]
    dev_colors = ["#2ca02c" if d >= 0 else "#e45756" for d in deviations]

    if not many_months:
        # Compact vertical layout for up to 16 months.
        positions = np.arange(count)
        bars = ax1.bar(
            positions, rainfall, width=0.62, color=bar_colors,
            edgecolor="black", alpha=0.85, label="Monthly Rainfall",
        )
        ax1.set_xticks(positions)
        ax1.set_xticklabels(months, rotation=30 if count > 10 else 0,
                             ha="right" if count > 10 else "center")
        ax1.axhline(mean_rain, color="green", linestyle="--", linewidth=1.8,
                    label=f"Mean/Avg ({mean_rain:.2f} mm)")
        ax1.axhline(m["median"], color="purple", linestyle="-.", linewidth=1.4,
                    label=f"Median ({m['median']:.2f} mm)")
        ax1.axhline(200, color="red", linestyle=":", linewidth=1.5,
                    label="Heavy Threshold (200 mm)")
        ax1.axhspan(
            mean_rain - m["std"], mean_rain + m["std"],
            color="green", alpha=0.12,
            label=f"±1 Std Dev Band ({m['std']:.2f} mm)",
        )
        ax1.set_title("Monthly Rainfall")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Rainfall (mm)")

        max_rain = max(float(np.max(rainfall)), 1.0)
        ax1.set_ylim(0, max_rain * 1.18)
        # Use a reserved side strip inside the axes; bars stay at their natural scale.
        ax1.set_xlim(-0.55, count + 2.35)
        ax1.legend(
            loc="upper left", bbox_to_anchor=(count + 0.05, max_rain * 1.15),
            bbox_transform=ax1.transData, fontsize=7, borderaxespad=0.2,
        )
        ax1.text(
            count + 1.15, max_rain * 0.48, build_stats_box_text(m),
            fontsize=7.5, verticalalignment="center", horizontalalignment="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8f9fa",
                  "edgecolor": "#888888", "alpha": 0.97},
            zorder=6,
        )
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            tag = ""
            if count > 1:
                if idx == m["max_idx"]:
                    tag = " (Max)"
                elif idx == m["min_idx"]:
                    tag = " (Min)"
            ax1.annotate(
                f"{format_num(height)}{tag}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if tag else "normal", clip_on=False,
            )
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        # Long datasets: horizontal bars, with an information strip above them.
        positions = np.arange(count)
        bars = ax1.barh(
            positions, rainfall, height=0.68, color=bar_colors,
            edgecolor="black", alpha=0.85, label="Monthly Rainfall",
        )
        ax1.axvline(mean_rain, color="green", linestyle="--", linewidth=1.6,
                    label=f"Mean/Avg ({mean_rain:.2f} mm)")
        ax1.axvline(m["median"], color="purple", linestyle="-.", linewidth=1.3,
                    label=f"Median ({m['median']:.2f} mm)")
        ax1.axvline(200, color="red", linestyle=":", linewidth=1.4,
                    label="Heavy Threshold (200 mm)")
        ax1.axvspan(
            mean_rain - m["std"], mean_rain + m["std"],
            color="green", alpha=0.12,
            label=f"±1 Std Dev Band ({m['std']:.2f} mm)",
        )
        ax1.set_yticks(positions)
        ax1.set_yticklabels(months, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_ylim(count - 0.5, -4.4)
        max_rain = max(float(np.max(rainfall)), 1.0)
        ax1.set_xlim(0, max_rain * 1.18)
        ax1.set_title(f"Monthly Rainfall ({count} months)")
        ax1.set_xlabel("Rainfall (mm)")
        ax1.set_ylabel("Month")
        ax1.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99),
                   fontsize=6.5, ncol=1, borderaxespad=0.2)
        ax1.text(
            0.99, 0.99, build_stats_box_text(m),
            transform=ax1.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8f9fa",
                  "edgecolor": "#888888", "alpha": 0.97},
            zorder=6,
        )
        for idx, bar in enumerate(bars):
            width = bar.get_width()
            tag = ""
            if count > 1:
                if idx == m["max_idx"]:
                    tag = " (Max)"
                elif idx == m["min_idx"]:
                    tag = " (Min)"
            label = f"{format_num(width)}{tag}"
            ax1.annotate(
                label, (width, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7,
                fontweight="bold" if tag else "normal", clip_on=False,
            )
        ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Deviation chart is horizontal for both layouts, matching month order.
    dev_bars = ax2.barh(months, deviations, color=dev_colors, edgecolor="black", alpha=0.85)
    ax2.axvline(0, color="green", linewidth=1.6,
                label=f"Mean Baseline ({mean_rain:.2f} mm)")
    if m["std"] > 0:
        ax2.axvline(m["std"], color="orange", linestyle="--", linewidth=1.3,
                    label=f"+1 Std Dev (+{m['std']:.2f} mm)")
        ax2.axvline(-m["std"], color="orange", linestyle="--", linewidth=1.3,
                    label=f"-1 Std Dev (-{m['std']:.2f} mm)")
    ax2.set_title("Deviation from Average (Above vs Below Mean)")
    ax2.set_xlabel("Difference from Mean (mm)")
    ax2.set_ylabel("Month")
    ax2.tick_params(axis="y", labelsize=8 if many_months else 9)
    low = min(float(np.min(deviations)), -float(m["std"]), 0.0)
    high = max(float(np.max(deviations)), float(m["std"]), 0.0)
    span = max(high - low, 1.0)
    ax2.set_xlim(low - 0.16 * span, high + 0.24 * span)
    ax2.legend(loc="best", fontsize=7.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    # Place labels inside bars when there is room; otherwise just beyond the
    # bar endpoint, with axis padding reserved to avoid clipping.
    inside_threshold = max(span * 0.12, 1.0)
    for bar in dev_bars:
        width = bar.get_width()
        y_mid = bar.get_y() + bar.get_height() / 2
        if width < 0 and abs(width) >= inside_threshold:
            ax2.annotate(
                f"{width:+.2f} mm", (width, y_mid),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.5,
                fontweight="bold", color="white",
            )
        else:
            offset = 4 if width >= 0 else -4
            ax2.annotate(
                f"{width:+.2f} mm", (width, y_mid),
                xytext=(offset, 0), textcoords="offset points",
                ha="left" if width >= 0 else "right",
                va="center", fontsize=7.5, fontweight="bold",
                clip_on=False,
            )

    if many_months:
        fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.07, wspace=0.28)
    else:
        fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.13, wspace=0.22)

    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    plt.close(fig)

def main():
    """Interactive menu with support for selecting CSV files by name or full path."""
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "rain.csv"

    if csv_path.exists():
        df = load_rainfall_csv(str(csv_path))
        last_mtime = csv_path.stat().st_mtime
        print(f"CSV file opened: {csv_path.name}")
    else:
        df = pd.DataFrame(columns=["Month", "Rainfall_mm"])
        last_mtime = None
        print("Default file 'rain.csv' was not found. Use option 4 to open a CSV file.")

    while True:
        print("\n--- Rainfall Data Analyzer Menu ---")
        print("1. Statistics")
        print("2. Plot rainfall trend")
        print("3. Compare with average and plot")
        print("4. Open another CSV file")
        print("5. Exit")

        choice = input("Enter menu option (1-5): ").strip()

        # Reload the currently selected CSV silently if it changes on disk.
        if csv_path.exists():
            current_mtime = csv_path.stat().st_mtime
            if current_mtime != last_mtime:
                df = load_rainfall_csv(str(csv_path))
                last_mtime = current_mtime

        if choice == "1":
            analyze_statistics(df)
        elif choice == "2":
            plot_rainfall_trend(df)
        elif choice == "3":
            compare_with_average_and_plot(df)
        elif choice == "4":
            print("\n--- Open Another CSV File ---")
            entered_path = input(
                "Enter CSV filename (if in project folder) OR full file path: "
            ).strip().strip('"').strip("'")

            if not entered_path:
                print("No filename or path entered. Keeping the current CSV file.")
                continue

            selected_path = Path(entered_path).expanduser()

            # A relative filename is searched in the project folder.
            # An absolute path is used directly.
            if not selected_path.is_absolute():
                selected_path = project_dir / selected_path

            selected_path = selected_path.resolve()

            if not selected_path.is_file():
                print(f"Error: CSV file not found: {selected_path}")
                continue

            if selected_path.suffix.lower() != ".csv":
                print("Error: Please select a file with the .csv extension.")
                continue

            # Switch to the selected file even when every row is invalid.
            # This is important for validation test cases: the user must be
            # able to inspect the selected CSV's errors instead of silently
            # continuing to analyze the previously selected file.
            new_df = load_rainfall_csv(str(selected_path))

            csv_path = selected_path
            df = new_df
            last_mtime = selected_path.stat().st_mtime

            if new_df.empty:
                print(
                    f"CSV file opened: {selected_path} "
                    "(no valid rainfall records; the selected file is now active)"
                )
            else:
                print(f"CSV file opened successfully: {selected_path}")

        elif choice == "5":
            print("Exiting Rainfall Data Analyzer.")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()