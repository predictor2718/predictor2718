"""Generate an animated SVG of Brownian motion paths.

Simulates independent Brownian motion paths and renders them as an SVG
with CSS stroke-dashoffset animation, so the paths appear to draw themselves.
Each run produces different random paths.
"""

import numpy as np


def sim_brownian_motion(T: float, n_steps: int, M: int) -> np.ndarray:
    """Simulate M independent Brownian motion paths on [0, T]."""
    dt = T / n_steps
    increments = np.random.normal(0, np.sqrt(dt), size=(n_steps, M))
    bm = np.zeros((n_steps + 1, M))
    bm[1:, :] = np.cumsum(increments, axis=0)
    return bm


def path_to_svg_d(xs: np.ndarray, ys: np.ndarray) -> str:
    """Convert x/y arrays to an SVG path d attribute."""
    points = [f"M {xs[0]:.2f} {ys[0]:.2f}"]
    for x, y in zip(xs[1:], ys[1:]):
        points.append(f"L {x:.2f} {y:.2f}")
    return " ".join(points)


def generate_svg(
    n_paths: int = 5,
    n_steps: int = 200,
    T: float = 1.0,
    width: int = 800,
    height: int = 300,
) -> str:
    """Generate an animated Brownian motion SVG."""
    bm = sim_brownian_motion(T, n_steps, n_paths)
    t = np.linspace(0, T, n_steps + 1)

    # Layout
    pad_left, pad_right, pad_top, pad_bottom = 50, 20, 40, 40
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Scale y to fit all paths with some breathing room
    y_min, y_max = bm.min(), bm.max()
    y_range = y_max - y_min if y_max - y_min > 0.01 else 1.0
    y_margin = y_range * 0.15
    y_min -= y_margin
    y_max += y_margin
    y_range = y_max - y_min

    def scale_x(val):
        return pad_left + (val / T) * plot_w

    def scale_y(val):
        return pad_top + (1 - (val - y_min) / y_range) * plot_h

    colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7"]

    # Pre-compute path lengths for each path (needed for stroke animation)
    path_lengths = []
    for i in range(n_paths):
        xs = np.array([scale_x(tv) for tv in t])
        ys = np.array([scale_y(bm[j, i]) for j in range(n_steps + 1)])
        dx = np.diff(xs)
        dy = np.diff(ys)
        path_lengths.append(int(np.sum(np.sqrt(dx**2 + dy**2))) + 10)

    # Build SVG
    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
    )

    # Styles — use hardcoded dasharray values (GitHub strips CSS custom properties)
    lines.append("<style>")
    lines.append(
        ".axis-line { stroke: #6b7280; stroke-width: 0.5; }"
    )
    lines.append(
        ".axis-text { font-family: monospace; font-size: 11px; fill: #9ca3af; }"
    )
    lines.append(
        ".title-text { font-family: monospace; font-size: 14px; "
        "fill: #d1d5db; font-weight: bold; }"
    )
    lines.append(
        ".grid-line { stroke: #374151; stroke-width: 0.3; stroke-dasharray: 4 2; }"
    )
    for i in range(n_paths):
        pl = path_lengths[i]
        lines.append(
            f".path-{i} {{"
            f" fill: none; stroke: {colors[i % len(colors)]};"
            f" stroke-width: 2; stroke-linecap: round;"
            f" opacity: 0.85;"
            f" animation: draw-{i} 3s ease-out {i * 0.4}s forwards;"
            f" stroke-dasharray: {pl};"
            f" stroke-dashoffset: {pl};"
            f"}}"
        )
        lines.append(
            f"@keyframes draw-{i} {{"
            f" to {{ stroke-dashoffset: 0; }}"
            f"}}"
        )
    lines.append("</style>")

    # Background
    lines.append(
        f'<rect width="{width}" height="{height}" rx="8" '
        f'fill="#0d1117" />'
    )

    # Grid lines (horizontal)
    n_grid = 5
    for i in range(n_grid + 1):
        y_val = y_min + i * y_range / n_grid
        sy = scale_y(y_val)
        lines.append(
            f'<line x1="{pad_left}" y1="{sy:.1f}" '
            f'x2="{width - pad_right}" y2="{sy:.1f}" class="grid-line" />'
        )
        lines.append(
            f'<text x="{pad_left - 6}" y="{sy + 4:.1f}" '
            f'text-anchor="end" class="axis-text">{y_val:.1f}</text>'
        )

    # Grid lines (vertical)
    for i in range(6):
        t_val = i * T / 5
        sx = scale_x(t_val)
        lines.append(
            f'<line x1="{sx:.1f}" y1="{pad_top}" '
            f'x2="{sx:.1f}" y2="{height - pad_bottom}" class="grid-line" />'
        )

    # Axis lines
    lines.append(
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" '
        f'x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis-line" />'
    )
    lines.append(
        f'<line x1="{pad_left}" y1="{pad_top}" '
        f'x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line" />'
    )

    # Axis labels
    lines.append(
        f'<text x="{width / 2}" y="{height - 5}" '
        f'text-anchor="middle" class="axis-text">t</text>'
    )
    lines.append(
        f'<text x="12" y="{height / 2}" '
        f'text-anchor="middle" class="axis-text" '
        f'transform="rotate(-90 12 {height / 2})">W(t)</text>'
    )

    # Title
    lines.append(
        f'<text x="{width / 2}" y="22" '
        f'text-anchor="middle" class="title-text">Brownian Motion</text>'
    )

    # Brownian motion paths
    for i in range(n_paths):
        xs = np.array([scale_x(tv) for tv in t])
        ys = np.array([scale_y(bm[j, i]) for j in range(n_steps + 1)])
        d = path_to_svg_d(xs, ys)
        lines.append(f'<path d="{d}" class="path-{i}" />')

    # Zero line (W=0 reference)
    zero_y = scale_y(0)
    if pad_top < zero_y < height - pad_bottom:
        lines.append(
            f'<line x1="{pad_left}" y1="{zero_y:.1f}" '
            f'x2="{width - pad_right}" y2="{zero_y:.1f}" '
            f'stroke="#4b5563" stroke-width="0.8" stroke-dasharray="6 3" />'
        )

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    svg = generate_svg()
    with open("brownian_motion.svg", "w") as f:
        f.write(svg)
    print("Generated brownian_motion.svg")
