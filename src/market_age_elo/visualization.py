from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_NANOSECONDS_PER_DAY = 86_400_000_000_000.0


def _get_matplotlib_pyplot():
    import os
    import tempfile
    from pathlib import Path

    if not os.environ.get("MPLCONFIGDIR"):
        mpl_dir = Path(tempfile.gettempdir()) / "player_elo_mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    if not os.environ.get("XDG_CACHE_HOME"):
        xdg_dir = Path(tempfile.gettempdir()) / "player_elo_xdg_cache"
        xdg_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_dir)

    import matplotlib

    if os.environ.get("DISPLAY", "") == "" and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def load_player_timeline_data(input_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "match_date" not in df.columns:
        raise KeyError("Input data must include `match_date`.")
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")
    df = df[df["match_date"].notna()].copy()
    return df.sort_values(["match_date", "match_id", "player_id"]).reset_index(drop=True)


def select_player_timeseries(
    df: pd.DataFrame,
    player_id: Optional[int] = None,
    player_name: Optional[str] = None,
) -> pd.DataFrame:
    if player_id is None and not player_name:
        raise ValueError("Provide either `player_id` or `player_name`.")

    out = df.copy()
    if player_id is not None:
        out = out[pd.to_numeric(out.get("player_id"), errors="coerce") == int(player_id)]
    if player_name:
        name_norm = str(player_name).strip().lower()
        out = out[out.get("player_name", "").astype(str).str.lower() == name_norm]

    out = out.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No rows found for the selected player in the provided data.")
    return out


def _player_ranking_series(df: pd.DataFrame, ranking_col: str, smooth_window: int) -> tuple[pd.Series, pd.Series, int]:
    if ranking_col not in df.columns:
        raise KeyError(f"Missing requested ranking column: {ranking_col}")
    window = max(int(smooth_window), 1)
    raw = pd.to_numeric(df[ranking_col], errors="coerce")
    smooth = raw.rolling(window=window, min_periods=1).mean()
    return raw, smooth, window


def _residual_series(df: pd.DataFrame, residual_col: str) -> pd.Series:
    if residual_col not in df.columns:
        raise KeyError(f"Missing requested residual column: {residual_col}")
    return pd.to_numeric(df[residual_col], errors="coerce")


def _smoothing_spline_series(
    x: pd.Series,
    y: pd.Series,
    spline_s: Optional[float] = None,
    spline_strength: float = 0.75,
    min_points: int = 4,
) -> tuple[pd.Series, bool]:
    x_dt = pd.to_datetime(x, utc=True, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    out = y_num.copy()

    valid_mask = x_dt.notna() & y_num.notna()
    if int(valid_mask.sum()) < int(min_points):
        return out, False

    x_valid = x_dt.loc[valid_mask]
    y_valid = y_num.loc[valid_mask]

    # UnivariateSpline is most stable with strictly increasing x;
    # aggregate duplicate timestamps before fitting.
    fit_df = pd.DataFrame(
        {
            "x_ns": x_valid.astype("int64").to_numpy(dtype="float64"),
            "y": y_valid.to_numpy(dtype="float64"),
        }
    )
    fit_df = fit_df.groupby("x_ns", as_index=False)["y"].mean().sort_values("x_ns")
    if fit_df.shape[0] < int(min_points):
        return out, False

    x_fit = fit_df["x_ns"].to_numpy(dtype="float64")
    y_fit = fit_df["y"].to_numpy(dtype="float64")
    x_base = float(x_fit.min())
    x_fit_days = (x_fit - x_base) / _NANOSECONDS_PER_DAY
    if np.ptp(x_fit_days) <= 0.0:
        return out, False

    spline_order = max(1, min(3, len(x_fit_days) - 1))
    if spline_s is None:
        variance = float(np.nanvar(y_fit))
        if not np.isfinite(variance) or variance <= 0.0:
            smooth_s = float(len(y_fit))
        else:
            smooth_s = float(max(1e-8, float(spline_strength) * len(y_fit) * variance))
    else:
        smooth_s = float(spline_s)

    try:
        from scipy.interpolate import UnivariateSpline

        spline = UnivariateSpline(x_fit_days, y_fit, k=spline_order, s=smooth_s)
    except Exception:  # pragma: no cover - optional dependency/runtime issues
        return out, False

    x_eval_days = (x_valid.astype("int64").to_numpy(dtype="float64") - x_base) / _NANOSECONDS_PER_DAY
    y_smooth = pd.Series(spline(x_eval_days), index=y_valid.index, dtype="float64")
    out.loc[y_smooth.index] = y_smooth
    return out, True


def _optional_baseline_ranking_series(
    baseline_player_df: Optional[pd.DataFrame],
    baseline_col: str,
    smooth_window: int,
) -> tuple[Optional[pd.Series], Optional[pd.Series], int]:
    if baseline_player_df is None:
        return None, None, max(int(smooth_window), 1)
    if "match_date" not in baseline_player_df.columns:
        raise KeyError("Baseline player data must include `match_date`.")

    baseline_df = baseline_player_df.copy().sort_values("match_date")
    baseline_df["match_date"] = pd.to_datetime(baseline_df["match_date"], utc=True, errors="coerce")
    baseline_df = baseline_df[baseline_df["match_date"].notna()].copy()
    if baseline_df.empty:
        return None, None, max(int(smooth_window), 1)

    _baseline_raw, baseline_smooth, window = _player_ranking_series(baseline_df, baseline_col, smooth_window)
    return baseline_df["match_date"], baseline_smooth, window


def plot_player_elo_timeline(
    player_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    elo_col: str = "player_elo_post",
    smooth_window: int = 5,
    residual_col: str = "performance_residual",
    include_residual: bool = True,
    include_age: bool = True,
    include_market_value: bool = True,
    use_subplots: bool = True,
    baseline_player_df: Optional[pd.DataFrame] = None,
    baseline_elo_col: str = "player_elo_post",
    baseline_label: str = "Baseline Ranking",
    residual_spline: bool = True,
    residual_spline_s: Optional[float] = None,
    residual_spline_strength: float = 0.75,
    show: bool = False,
):
    plt = _get_matplotlib_pyplot()
    if "match_date" not in player_df.columns:
        raise KeyError("Missing required column: match_date")

    df = player_df.copy().sort_values("match_date")
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")
    df = df[df["match_date"].notna()].copy()
    if df.empty:
        raise ValueError("Player data has no valid `match_date` values.")

    x = df["match_date"]
    player_ranking_raw, player_ranking_smooth, window = _player_ranking_series(df, elo_col, smooth_window)
    residual = _residual_series(df, residual_col) if include_residual else None
    residual_for_plot = residual
    residual_label = "Performance Residual"
    if include_residual and residual is not None and residual_spline:
        residual_for_plot, spline_used = _smoothing_spline_series(
            x=x,
            y=residual,
            spline_s=residual_spline_s,
            spline_strength=residual_spline_strength,
        )
        if spline_used:
            residual_label = "Performance Residual (Spline)"

    baseline_x, baseline_ranking_smooth, baseline_window = _optional_baseline_ranking_series(
        baseline_player_df=baseline_player_df,
        baseline_col=baseline_elo_col,
        smooth_window=smooth_window,
    )

    player_label = str(df.get("player_name", pd.Series(["Unknown"])).iloc[0])
    player_id_label = str(df.get("player_id", pd.Series(["?"])).iloc[0])

    if use_subplots:
        n_panels = 1 + int(include_residual) + int(include_market_value) + int(include_age)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        idx = 0
        ax_rank = axes[idx]
        idx += 1
        ax_rank.plot(x, player_ranking_raw, color="tab:blue", linewidth=1.7, label="Player Ranking (Raw)")
        ax_rank.plot(
            x,
            player_ranking_smooth,
            color="tab:purple",
            linewidth=2.4,
            label=f"Player Ranking (Smoothed, window={window})",
        )
        if baseline_x is not None and baseline_ranking_smooth is not None:
            ax_rank.plot(
                baseline_x,
                baseline_ranking_smooth,
                color="#4d4d4d",
                linewidth=2.2,
                linestyle="--",
                label=f"{baseline_label} (Smoothed, window={baseline_window})",
            )
        ax_rank.set_ylabel("Player Ranking")
        ax_rank.grid(alpha=0.25)
        ax_rank.set_title(f"{player_label} (ID {player_id_label}) - Player Ranking Timeline")
        ax_rank.legend(loc="upper left")

        if include_residual:
            ax_res = axes[idx]
            idx += 1
            ax_res.plot(x, residual_for_plot, color="tab:red", linewidth=1.8, label=residual_label)
            ax_res.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
            ax_res.set_ylabel("Residual")
            ax_res.grid(alpha=0.25)
            ax_res.legend(loc="upper left")

        if include_market_value:
            if "market_value" not in df.columns:
                raise KeyError("`include_market_value=True` requires `market_value`.")
            ax_mv = axes[idx]
            idx += 1
            mv_m = pd.to_numeric(df["market_value"], errors="coerce") / 1_000_000.0
            ax_mv.plot(x, mv_m, color="tab:orange", linewidth=1.8)
            ax_mv.set_ylabel("MV (M)")
            ax_mv.grid(alpha=0.25)

        # Keep age on the lowest panel when subplots are enabled.
        if include_age:
            if "player_age_years" not in df.columns:
                raise KeyError("`include_age=True` requires `player_age_years`.")
            ax_age = axes[idx]
            ax_age.plot(x, df["player_age_years"], color="tab:green", linewidth=1.8)
            ax_age.set_ylabel("Age")
            ax_age.grid(alpha=0.25)

        axes[-1].set_xlabel("Match Date")
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5.2))
        h_raw, = ax1.plot(x, player_ranking_raw, color="tab:blue", linewidth=1.7, label="Player Ranking (Raw)")
        h_smooth, = ax1.plot(
            x,
            player_ranking_smooth,
            color="tab:purple",
            linewidth=2.4,
            label=f"Player Ranking (Smoothed, window={window})",
        )
        if baseline_x is not None and baseline_ranking_smooth is not None:
            h_baseline, = ax1.plot(
                baseline_x,
                baseline_ranking_smooth,
                color="#4d4d4d",
                linewidth=2.2,
                linestyle="--",
                label=f"{baseline_label} (Smoothed, window={baseline_window})",
            )
        ax1.set_ylabel("Player Ranking", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.25)
        ax1.set_title(f"{player_label} (ID {player_id_label}) - Player Ranking Timeline")
        line_handles = [h_raw, h_smooth]
        line_labels = ["Player Ranking (Raw)", f"Player Ranking (Smoothed, window={window})"]
        if baseline_x is not None and baseline_ranking_smooth is not None:
            line_handles.append(h_baseline)
            line_labels.append(f"{baseline_label} (Smoothed, window={baseline_window})")

        axis_offset = 1.0
        if include_residual:
            ax_res = ax1.twinx()
            h_res, = ax_res.plot(x, residual_for_plot, color="tab:red", linewidth=1.6)
            ax_res.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
            ax_res.set_ylabel("Residual", color="tab:red")
            ax_res.tick_params(axis="y", labelcolor="tab:red")
            line_handles.append(h_res)
            line_labels.append(residual_label)
            axis_offset += 0.10

        if include_age:
            if "player_age_years" not in df.columns:
                raise KeyError("`include_age=True` requires `player_age_years`.")
            ax_age = ax1.twinx()
            ax_age.spines["right"].set_position(("axes", axis_offset))
            h_age, = ax_age.plot(x, df["player_age_years"], color="tab:green", linewidth=1.6)
            ax_age.set_ylabel("Age", color="tab:green")
            ax_age.tick_params(axis="y", labelcolor="tab:green")
            line_handles.append(h_age)
            line_labels.append("Age")
            axis_offset += 0.10

        if include_market_value:
            if "market_value" not in df.columns:
                raise KeyError("`include_market_value=True` requires `market_value`.")
            ax_mv = ax1.twinx()
            ax_mv.spines["right"].set_position(("axes", axis_offset))
            mv_m = pd.to_numeric(df["market_value"], errors="coerce") / 1_000_000.0
            h_mv, = ax_mv.plot(x, mv_m, color="tab:orange", linewidth=1.6)
            ax_mv.set_ylabel("MV (M)", color="tab:orange")
            ax_mv.tick_params(axis="y", labelcolor="tab:orange")
            line_handles.append(h_mv)
            line_labels.append("Market Value (M)")

        ax1.set_xlabel("Match Date")
        ax1.legend(line_handles, line_labels, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_player_elo_timeline_interactive(
    player_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    elo_col: str = "player_elo_post",
    smooth_window: int = 5,
    residual_col: str = "performance_residual",
    include_residual: bool = True,
    include_age: bool = True,
    include_market_value: bool = True,
    use_subplots: bool = True,
    baseline_player_df: Optional[pd.DataFrame] = None,
    baseline_elo_col: str = "player_elo_post",
    baseline_label: str = "Baseline Ranking",
    residual_spline: bool = True,
    residual_spline_s: Optional[float] = None,
    residual_spline_strength: float = 0.75,
    show: bool = False,
) -> go.Figure:
    if "match_date" not in player_df.columns:
        raise KeyError("Missing required column: match_date")

    df = player_df.copy().sort_values("match_date")
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")
    df = df[df["match_date"].notna()].copy()
    if df.empty:
        raise ValueError("Player data has no valid `match_date` values.")

    x = df["match_date"]
    player_ranking_raw, player_ranking_smooth, window = _player_ranking_series(df, elo_col, smooth_window)
    residual = _residual_series(df, residual_col) if include_residual else None
    residual_for_plot = residual
    residual_label = "Performance Residual"
    if include_residual and residual is not None and residual_spline:
        residual_for_plot, spline_used = _smoothing_spline_series(
            x=x,
            y=residual,
            spline_s=residual_spline_s,
            spline_strength=residual_spline_strength,
        )
        if spline_used:
            residual_label = "Performance Residual (Spline)"

    baseline_x, baseline_ranking_smooth, baseline_window = _optional_baseline_ranking_series(
        baseline_player_df=baseline_player_df,
        baseline_col=baseline_elo_col,
        smooth_window=smooth_window,
    )
    player_label = str(df.get("player_name", pd.Series(["Unknown"])).iloc[0])
    player_id_label = str(df.get("player_id", pd.Series(["?"])).iloc[0])

    if use_subplots:
        rows = 1 + int(include_residual) + int(include_market_value) + int(include_age)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        row = 1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=player_ranking_raw,
                mode="lines",
                name="Player Ranking (Raw)",
                line=dict(color="#1f77b4", width=1.8),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=player_ranking_smooth,
                mode="lines",
                name=f"Player Ranking (Smoothed, window={window})",
                line=dict(color="#9467bd", width=2.6),
            ),
            row=row,
            col=1,
        )
        if baseline_x is not None and baseline_ranking_smooth is not None:
            fig.add_trace(
                go.Scatter(
                    x=baseline_x,
                    y=baseline_ranking_smooth,
                    mode="lines",
                    name=f"{baseline_label} (Smoothed, window={baseline_window})",
                    line=dict(color="#4d4d4d", width=2.4, dash="dash"),
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text="Player Ranking", row=row, col=1)
        row += 1

        if include_residual:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=residual_for_plot,
                    mode="lines",
                    name=residual_label,
                    line=dict(color="#d62728", width=1.9),
                ),
                row=row,
                col=1,
            )
            fig.add_hline(y=0.0, line_dash="dash", line_color="#999999", row=row, col=1)
            fig.update_yaxes(title_text="Residual", row=row, col=1)
            row += 1

        if include_market_value:
            if "market_value" not in df.columns:
                raise KeyError("`include_market_value=True` requires `market_value`.")
            mv_m = pd.to_numeric(df["market_value"], errors="coerce") / 1_000_000.0
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mv_m,
                    mode="lines",
                    name="Market Value (M)",
                    line=dict(color="#ff7f0e", width=2.0),
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="MV (M)", row=row, col=1)
            row += 1

        # Keep age on the lowest panel when subplots are enabled.
        if include_age:
            if "player_age_years" not in df.columns:
                raise KeyError("`include_age=True` requires `player_age_years`.")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["player_age_years"],
                    mode="lines",
                    name="Age",
                    line=dict(color="#2ca02c", width=2.0),
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="Age", row=row, col=1)
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=player_ranking_raw,
                mode="lines",
                name="Player Ranking (Raw)",
                line=dict(color="#1f77b4", width=1.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=player_ranking_smooth,
                mode="lines",
                name=f"Player Ranking (Smoothed, window={window})",
                line=dict(color="#9467bd", width=2.6),
            )
        )
        if baseline_x is not None and baseline_ranking_smooth is not None:
            fig.add_trace(
                go.Scatter(
                    x=baseline_x,
                    y=baseline_ranking_smooth,
                    mode="lines",
                    name=f"{baseline_label} (Smoothed, window={baseline_window})",
                    line=dict(color="#4d4d4d", width=2.4, dash="dash"),
                )
            )

        overlays: list[tuple[str, pd.Series, str, str]] = []
        if include_residual:
            overlays.append((residual_label, residual_for_plot, "Residual", "#d62728"))
        if include_market_value:
            if "market_value" not in df.columns:
                raise KeyError("`include_market_value=True` requires `market_value`.")
            mv_m = pd.to_numeric(df["market_value"], errors="coerce") / 1_000_000.0
            overlays.append(("Market Value (M)", mv_m, "MV (M)", "#ff7f0e"))
        if include_age:
            if "player_age_years" not in df.columns:
                raise KeyError("`include_age=True` requires `player_age_years`.")
            overlays.append(("Age", pd.to_numeric(df["player_age_years"], errors="coerce"), "Age", "#2ca02c"))

        for i, (trace_name, series, _axis_title, color) in enumerate(overlays, start=2):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=series,
                    mode="lines",
                    name=trace_name,
                    yaxis=f"y{i}",
                    line=dict(color=color, width=1.8),
                )
            )

        layout: dict[str, object] = {
            "yaxis": {"title": "Player Ranking"},
            "legend": {"orientation": "h", "x": 0.0, "y": 1.08},
        }
        right_count = 0
        left_count = 0
        for i, (_trace_name, _series, axis_title, _color) in enumerate(overlays, start=2):
            side = "right" if i % 2 == 0 else "left"
            cfg = {"title": axis_title, "overlaying": "y", "side": side, "anchor": "free"}
            if side == "right":
                cfg["position"] = max(0.70, 1.0 - 0.08 * right_count)
                right_count += 1
            else:
                cfg["position"] = min(0.30, 0.00 + 0.08 * left_count)
                left_count += 1
            layout[f"yaxis{i}"] = cfg
        fig.update_layout(**layout)

    fig.update_layout(
        template="plotly_white",
        title=f"{player_label} (ID {player_id_label}) - Player Ranking Timeline",
        xaxis_title="Match Date",
        height=360 + (220 * (int(include_age) + int(include_market_value))) if use_subplots else 520,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix in {".html", ""}:
            fig.write_html(out if suffix else out.with_suffix(".html"))
        else:
            try:
                fig.write_image(out)
            except Exception as exc:  # pragma: no cover - depends on optional plotly image backend
                raise RuntimeError(
                    "Static image export for Plotly requires kaleido. "
                    "Install kaleido or use an .html output path."
                ) from exc

    if show:
        fig.show()

    return fig
