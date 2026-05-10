"""Plotly layout defaults aligned with Executive Overview (DM Sans palette, slate canvas)."""

from __future__ import annotations

from typing import Any

_DSS_FONT = "DM Sans, Arial, sans-serif"
DSS_TITLE_FONT = dict(family=_DSS_FONT, size=15, color="#0f172a")


def dss_title(text: str) -> dict[str, Any]:
    return dict(text=text, font=DSS_TITLE_FONT)


def apply_dss_layout(
    fig: Any,
    *,
    height: int | None = 380,
    margin: dict[str, int] | None = None,
) -> Any:
    """Paper background, typography, margins. Safe for Indicator, Scattergeo (no axis overrides)."""
    mdef = dict(l=22, r=26, t=54, b=42)
    if margin:
        mdef = {**mdef, **margin}
    kw: dict[str, Any] = dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family=_DSS_FONT, size=13, color="#475569"),
        margin=mdef,
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family=_DSS_FONT),
    )
    if height is not None:
        kw["height"] = height
    return fig.update_layout(**kw)


def apply_dss_cartesian_grid(fig: Any) -> Any:
    """Primary x/y grids for simple charts; skip on figures with overlays if it misbehaves."""
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(226,232,240,0.95)",
        zeroline=False,
        zerolinewidth=0,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(226,232,240,0.95)",
        zeroline=False,
        zerolinewidth=0,
    )
    return fig
