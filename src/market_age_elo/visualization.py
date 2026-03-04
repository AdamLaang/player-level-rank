from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot_player_elo_timeline(
    player_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    elo_col: str = "player_elo_post",
    smooth_window: int = 5,
    include_age: bool = True,
    include_market_value: bool = True,
    use_subplots: bool = True,
    show: bool = False,
):
    plt = _get_matplotlib_pyplot()
    if "match_date" not in player_df.columns:
        raise KeyError("Missing required column: match_date")

    df = player_df.copy().sort_values("match_date")
    x = df["match_date"]
    player_ranking_raw, player_ranking_smooth, window = _player_ranking_series(df, elo_col, smooth_window)

    player_label = str(df.get("player_name", pd.Series(["Unknown"])).iloc[0])
    player_id_label = str(df.get("player_id", pd.Series(["?"])).iloc[0])

    if use_subplots:
        n_panels = 1 + int(include_age) + int(include_market_value)
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
        ax_rank.set_ylabel("Player Ranking")
        ax_rank.grid(alpha=0.25)
        ax_rank.set_title(f"{player_label} (ID {player_id_label}) - Player Ranking Timeline")
        ax_rank.legend(loc="upper left")

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
        ax1.set_ylabel("Player Ranking", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.25)
        ax1.set_title(f"{player_label} (ID {player_id_label}) - Player Ranking Timeline")
        line_handles = [h_raw, h_smooth]
        line_labels = ["Player Ranking (Raw)", f"Player Ranking (Smoothed, window={window})"]

        axis_offset = 1.0
        if include_age:
            if "player_age_years" not in df.columns:
                raise KeyError("`include_age=True` requires `player_age_years`.")
            ax_age = ax1.twinx()
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
    include_age: bool = True,
    include_market_value: bool = True,
    use_subplots: bool = True,
    show: bool = False,
) -> go.Figure:
    if "match_date" not in player_df.columns:
        raise KeyError("Missing required column: match_date")

    df = player_df.copy().sort_values("match_date")
    x = df["match_date"]
    player_ranking_raw, player_ranking_smooth, window = _player_ranking_series(df, elo_col, smooth_window)
    player_label = str(df.get("player_name", pd.Series(["Unknown"])).iloc[0])
    player_id_label = str(df.get("player_id", pd.Series(["?"])).iloc[0])

    if use_subplots:
        rows = 1 + int(include_age) + int(include_market_value)
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
        fig.update_yaxes(title_text="Player Ranking", row=row, col=1)
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

        if include_age:
            if "player_age_years" not in df.columns:
                raise KeyError("`include_age=True` requires `player_age_years`.")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["player_age_years"],
                    mode="lines",
                    name="Age",
                    yaxis="y2",
                    line=dict(color="#2ca02c", width=1.8),
                )
            )

        if include_market_value:
            if "market_value" not in df.columns:
                raise KeyError("`include_market_value=True` requires `market_value`.")
            mv_m = pd.to_numeric(df["market_value"], errors="coerce") / 1_000_000.0
            axis_name = "y3" if include_age else "y2"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mv_m,
                    mode="lines",
                    name="Market Value (M)",
                    yaxis=axis_name,
                    line=dict(color="#ff7f0e", width=1.8),
                )
            )

        layout = {
            "yaxis": {"title": "Player Ranking"},
            "yaxis2": {"title": "Age", "overlaying": "y", "side": "right"} if include_age else {},
            "legend": {"orientation": "h", "x": 0.0, "y": 1.08},
        }
        if include_market_value:
            if include_age:
                layout["yaxis3"] = {"title": "MV (M)", "anchor": "free", "overlaying": "y", "side": "left", "position": 0.0}
            else:
                layout["yaxis2"] = {"title": "MV (M)", "overlaying": "y", "side": "right"}
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
