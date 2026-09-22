"""Map helpers for the Streamlit dashboard."""

import numpy as np

CATEGORY_COLORS = {
    "Low": "#2ECC71",
    "Medium": "#F39C12",
    "High": "#E74C3C",
}

LEGEND_HTML = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background-color: rgba(255,255,255,0.96);
    padding: 14px;
    border: 2px solid #444;
    border-radius: 8px;
    font-size: 14px;
    color: #111;
    line-height: 1.6;
">
<b>Dominant demand category</b><br><br>
<span style="color:#2ECC71;">&#9679;</span> Low<br>
<span style="color:#F39C12;">&#9679;</span> Medium<br>
<span style="color:#E74C3C;">&#9679;</span> High<br><br>
<b>Bubble size</b><br>
Larger bubble = more rides
</div>
"""


def get_marker_color(category):
    return CATEGORY_COLORS.get(category, "#95A5A6")


def marker_radius(ride_count, max_rides, min_radius=10, max_radius=40):
    """
    Log-scaled radius.

    The old formula (8 + count ** 0.40, capped at 35) saturated the cap for
    almost every zone once counts were summed over a year, so every bubble
    came out the same size.
    """
    if max_rides <= 0:
        return min_radius

    scaled = np.log1p(max(ride_count, 0)) / np.log1p(max_rides)

    return float(min_radius + scaled * (max_radius - min_radius))
