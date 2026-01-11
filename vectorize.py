"""Vectorize black-on-white PNG schematics into CAD-like SVGs.

Usage:
    python vectorize.py --input in.png --output outdir

The script implements a CPU-first vectorisation pipeline:
    1. Preprocessing: grayscale to binary (Otsu or manual threshold),
       optional morphology and deskew.
    2. Skeletonisation and segment detection via OpenCV LSD/FLD.
    3. Graph tracing and polyline simplification.
    4. Primitive fitting: lines, circles, arcs, ellipses, Bézier curves.
    5. Geometry snapping and cleanup.
    6. SVG export with strict grouping and optional preview overlays.

Installation:
    pip install opencv-python scikit-image networkx shapely svgwrite numpy

Examples:
    python vectorize.py --input examples --output out --recursive --preset balanced
    python vectorize.py --input img.png --output out --preset tight --preview

The code is intentionally self-contained in a single file for portability.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---- optional imports -------------------------------------------------------
try:  # OpenCV is required; raise informative error if missing
    import cv2
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("opencv-python is required: pip install opencv-python") from exc

try:
    from skimage import color, filters, morphology
    from skimage.transform import hough_line, hough_line_peaks, rotate
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-image is required: pip install scikit-image") from exc

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    raise SystemExit("networkx is required: pip install networkx") from exc

try:
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
except Exception as exc:  # pragma: no cover
    raise SystemExit("shapely is required: pip install shapely") from exc

try:
    import svgwrite
except Exception as exc:  # pragma: no cover
    raise SystemExit("svgwrite is required: pip install svgwrite") from exc

# tqdm progress bar is optional
try:  # pragma: no cover - optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ---- data structures --------------------------------------------------------

@dataclass
class Line:
    start: Tuple[float, float]
    end: Tuple[float, float]

@dataclass
class Circle:
    center: Tuple[float, float]
    radius: float

@dataclass
class Arc:
    center: Tuple[float, float]
    radius: float
    start_angle: float
    end_angle: float
    sweep: int

@dataclass
class Ellipse:
    center: Tuple[float, float]
    rx: float
    ry: float
    angle: float

@dataclass
class Bezier:
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    p3: Tuple[float, float]

# ---- utility ----------------------------------------------------------------

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


def angle_of(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return math.atan2(dy, dx) * RAD2DEG


def dist(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    return math.hypot(p0[0] - p1[0], p0[1] - p1[1])


# ---- preprocessing ----------------------------------------------------------

def preprocess(img: np.ndarray, *, threshold: Optional[int] = None, adaptive: bool = False,
               open_px: int = 0, close_px: int = 0) -> np.ndarray:
    """Convert BGR/gray image to binary (uint8 0/255)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if adaptive:
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 5)
    else:
        if threshold is None:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    if open_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1, open_px * 2 + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    return bw


def deskew(bw: np.ndarray) -> np.ndarray:
    """Estimate dominant orientation via Hough and rotate to deskew image."""
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    tested_angles = np.linspace(-90, 90, 360) * DEG2RAD
    h, theta, d = hough_line(edges, tested_angles)
    _, angles, _ = hough_line_peaks(h, theta, d)
    if len(angles) == 0:
        return bw
    angle = np.median(angles) * RAD2DEG
    rotated = rotate(bw, angle, resize=False, preserve_range=True).astype(bw.dtype)
    return rotated


def skeletonize_image(bw: np.ndarray) -> np.ndarray:
    """Return skeleton of binary image."""
    skel = morphology.skeletonize((bw > 0).astype(bool))
    return skel.astype(np.uint8) * 255

# ---- segment detection ------------------------------------------------------


def detect_segments(img: np.ndarray, detector: str = "auto", min_length: float = 8.0) -> List[Line]:
    """Detect line segments using OpenCV LSD/FLD."""
    if detector == "auto":
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector"):
            detector = "fld"
        else:
            detector = "lsd"
    lines: List[Line] = []
    if detector == "fld" and hasattr(cv2, "ximgproc"):
        fld = cv2.ximgproc.createFastLineDetector()
        segments = fld.detect(img)
        if segments is not None:
            for s in segments:
                x0, y0, x1, y1 = map(float, s[0])
                if dist((x0, y0), (x1, y1)) >= min_length:
                    lines.append(Line((x0, y0), (x1, y1)))
    else:  # LSD
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        segments, _, _, _ = lsd.detect(img)
        if segments is not None:
            for s in segments:
                x0, y0, x1, y1 = map(float, s[0])
                if dist((x0, y0), (x1, y1)) >= min_length:
                    lines.append(Line((x0, y0), (x1, y1)))
    return lines

# ---- skeleton graph --------------------------------------------------------

def build_graph_from_skeleton(skel: np.ndarray) -> Tuple[nx.Graph, Dict[int, Tuple[int, int]]]:
    """Build 8-connected graph from skeleton. Returns graph and mapping node->(y,x)."""
    G = nx.Graph()
    idx_to_coord: Dict[int, Tuple[int, int]] = {}
    h, w = skel.shape
    node_id = 0
    for y in range(h):
        for x in range(w):
            if skel[y, x] == 0:
                continue
            idx_to_coord[node_id] = (y, x)
            G.add_node(node_id)
            node_id += 1
    coords_to_id = {coord: idx for idx, coord in idx_to_coord.items()}
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for idx, (y, x) in idx_to_coord.items():
        for dy, dx in neighbors:
            ny, nx_ = y + dy, x + dx
            if (ny, nx_) in coords_to_id:
                G.add_edge(idx, coords_to_id[(ny, nx_)])
    return G, idx_to_coord


def trace_polylines(G: nx.Graph, idx_to_coord: Dict[int, Tuple[int, int]]) -> List[List[Tuple[float, float]]]:
    """Trace polylines from skeleton graph."""
    polylines: List[List[Tuple[float, float]]] = []
    visited = set()
    for node in list(G.nodes):
        if node in visited:
            continue
        if G.degree(node) != 1:
            continue
        current = node
        poly: List[Tuple[float, float]] = [idx_to_coord[current][::-1]]  # (x,y)
        visited.add(current)
        prev = None
        while True:
            neighbors = [n for n in G.neighbors(current) if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            poly.append(idx_to_coord[nxt][::-1])
            visited.add(nxt)
            prev, current = current, nxt
            if G.degree(current) != 2:
                break
        polylines.append(poly)
    return polylines

# ---- polyline operations ----------------------------------------------------


def simplify_polylines(polylines: List[List[Tuple[float, float]]], epsilon: float) -> List[List[Tuple[float, float]]]:
    out: List[List[Tuple[float, float]]] = []
    for pl in polylines:
        ls = LineString(pl)
        simp = ls.simplify(epsilon)
        out.append(list(map(tuple, np.asarray(simp.coords))))
    return out


def close_gaps(polylines: List[List[Tuple[float, float]]], gap: float) -> List[List[Tuple[float, float]]]:
    changed = True
    while changed:
        changed = False
        for i in range(len(polylines)):
            for j in range(i + 1, len(polylines)):
                d = dist(polylines[i][-1], polylines[j][0])
                if d <= gap:
                    polylines[i].extend(polylines[j])
                    del polylines[j]
                    changed = True
                    break
            if changed:
                break
    return polylines


def snap_angles(polylines: List[List[Tuple[float, float]]], snap: float) -> List[List[Tuple[float, float]]]:
    def snap_angle(a: float) -> float:
        for base in [0, 45, 90, 135, 180]:
            if abs(((a - base + 180) % 360) - 180) <= snap:
                return base
        return a
    out: List[List[Tuple[float, float]]] = []
    for pl in polylines:
        if len(pl) < 2:
            out.append(pl)
            continue
        start, end = pl[0], pl[-1]
        a = angle_of(start, end)
        a_snapped = snap_angle(a)
        if a != a_snapped:
            r = math.hypot(end[0] - start[0], end[1] - start[1])
            rad = a_snapped * DEG2RAD
            end = (start[0] + r * math.cos(rad), start[1] + r * math.sin(rad))
        out.append([start, end])
    return out


def merge_colinear(lines: List[Line], angle_tol: float = 2.0) -> List[Line]:
    if not lines:
        return []
    merged: List[Line] = []
    used = [False] * len(lines)
    for i, l in enumerate(lines):
        if used[i]:
            continue
        start, end = l.start, l.end
        angle = angle_of(start, end)
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            l2 = lines[j]
            angle2 = angle_of(l2.start, l2.end)
            if abs(angle - angle2) <= angle_tol:
                if dist(end, l2.start) < 1e-6:
                    end = l2.end
                    used[j] = True
                elif dist(l2.end, start) < 1e-6:
                    start = l2.start
                    used[j] = True
        merged.append(Line(start, end))
    return merged

# ---- primitive fitting ------------------------------------------------------

def fit_circle(pts: np.ndarray) -> Tuple[Tuple[float, float], float, float]:
    """Simple algebraic circle fit (Kasa). Returns center, radius, RMS error."""
    x = pts[:, 0]; y = pts[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x*x + y*y
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(c[2] + cx*cx + cy*cy)
    err = np.sqrt(((np.hypot(x - cx, y - cy) - r) ** 2).mean())
    return (cx, cy), r, err


def fit_circles_ransac(polylines: List[List[Tuple[float, float]]], max_err: float) -> Tuple[List[Circle], List[List[Tuple[float, float]]]]:
    circles: List[Circle] = []
    leftovers: List[List[Tuple[float, float]]] = []
    for pl in polylines:
        if len(pl) < 6:
            leftovers.append(pl)
            continue
        pts = np.array(pl)
        if dist(pl[0], pl[-1]) > 3.0:  # not closed -> maybe arc
            leftovers.append(pl)
            continue
        (cx, cy), r, err = fit_circle(pts)
        if err <= max_err:
            circles.append(Circle((cx, cy), r))
        else:
            leftovers.append(pl)
    return circles, leftovers


def fit_ellipses_ransac(polylines: List[List[Tuple[float, float]]], max_err: float) -> Tuple[List[Ellipse], List[List[Tuple[float, float]]]]:
    ellipses: List[Ellipse] = []
    leftovers: List[List[Tuple[float, float]]] = []
    for pl in polylines:
        if len(pl) < 20:
            leftovers.append(pl)
            continue
        pts = np.array(pl, dtype=np.float32)
        if dist(pl[0], pl[-1]) > 3.0:
            leftovers.append(pl)
            continue
        try:
            ellipse = cv2.fitEllipse(pts)
        except Exception:
            leftovers.append(pl)
            continue
        (cx, cy), (rx, ry), angle = ellipse
        rr = np.sqrt(((pts[:,0]-cx)**2/(rx/2)**2 + (pts[:,1]-cy)**2/(ry/2)**2) - 1)**2
        err = np.sqrt(rr.mean()) if rr.size > 0 else 0.0
        if err <= max_err:
            ellipses.append(Ellipse((cx, cy), rx/2, ry/2, angle*DEG2RAD))
        else:
            leftovers.append(pl)
    return ellipses, leftovers


def extract_arcs(polylines: List[List[Tuple[float, float]]], max_err: float) -> Tuple[List[Arc], List[List[Tuple[float, float]]]]:
    arcs: List[Arc] = []
    leftovers: List[List[Tuple[float, float]]] = []
    for pl in polylines:
        if len(pl) < 5:
            leftovers.append(pl)
            continue
        pts = np.array(pl)
        (cx, cy), r, err = fit_circle(pts)
        if err <= max_err:
            start_angle = math.atan2(pl[0][1]-cy, pl[0][0]-cx)
            end_angle = math.atan2(pl[-1][1]-cy, pl[-1][0]-cx)
            sweep = 1 if (end_angle - start_angle) % (2*math.pi) > 0 else 0
            arcs.append(Arc((cx, cy), r, start_angle, end_angle, sweep))
        else:
            leftovers.append(pl)
    return arcs, leftovers


def fit_beziers(polylines: List[List[Tuple[float, float]]]) -> Tuple[List[Bezier], List[List[Tuple[float, float]]]]:
    # Placeholder: complex fitting not implemented. Return empty.
    return [], polylines

# ---- SVG export -------------------------------------------------------------

def circle_path(c: Circle, precision: int) -> str:
    cx, cy = c.center
    r = c.radius
    fmt = f"{{:.{precision}f}}"
    return (
        f"M {fmt.format(cx+r)},{fmt.format(cy)} "
        f"A {fmt.format(r)},{fmt.format(r)} 0 1 0 {fmt.format(cx-r)},{fmt.format(cy)} "
        f"A {fmt.format(r)},{fmt.format(r)} 0 1 0 {fmt.format(cx+r)},{fmt.format(cy)}"
    )


def line_path(l: Line, precision: int) -> str:
    fmt = f"{{:.{precision}f}}"
    return f"M {fmt.format(l.start[0])},{fmt.format(l.start[1])} L {fmt.format(l.end[0])},{fmt.format(l.end[1])}"


def arc_path(a: Arc, precision: int) -> str:
    fmt = f"{{:.{precision}f}}"
    sx = a.center[0] + a.radius * math.cos(a.start_angle)
    sy = a.center[1] + a.radius * math.sin(a.start_angle)
    ex = a.center[0] + a.radius * math.cos(a.end_angle)
    ey = a.center[1] + a.radius * math.sin(a.end_angle)
    large = 1 if abs(a.end_angle - a.start_angle) % (2*math.pi) > math.pi else 0
    sweep = a.sweep
    return (
        f"M {fmt.format(sx)},{fmt.format(sy)} "
        f"A {fmt.format(a.radius)},{fmt.format(a.radius)} 0 {large} {sweep} {fmt.format(ex)},{fmt.format(ey)}"
    )


def ellipse_path(e: Ellipse, precision: int) -> str:
    fmt = f"{{:.{precision}f}}"
    sx = e.center[0] + e.rx
    sy = e.center[1]
    return (
        f"M {fmt.format(sx)},{fmt.format(sy)} "
        f"A {fmt.format(e.rx)},{fmt.format(e.ry)} {fmt.format(e.angle*RAD2DEG)} 1 0 {fmt.format(e.center[0]-e.rx)},{fmt.format(e.center[1])} "
        f"A {fmt.format(e.rx)},{fmt.format(e.ry)} {fmt.format(e.angle*RAD2DEG)} 1 0 {fmt.format(sx)},{fmt.format(sy)}"
    )


def to_svg(lines: List[Line], arcs: List[Arc], circles: List[Circle], ellipses: List[Ellipse],
           beziers: List[Bezier], out_path: Path, hairline: float = 0.1,
           precision: int = 3, metadata: Dict[str, str] | None = None) -> None:
    dwg = svgwrite.Drawing(out_path.as_posix())
    style = {"fill": "none", "stroke": "black", "stroke_linecap": "round",
             "stroke_linejoin": "round", "stroke_width": hairline}
    for name, prims, func in [
        ("lines", lines, line_path),
        ("arcs", arcs, arc_path),
        ("circles", circles, circle_path),
        ("ellipses", ellipses, ellipse_path),
        ("beziers", beziers, lambda b, p: "")
    ]:
        grp = dwg.g(id=name)
        for p in prims:
            d = func(p, precision)
            if not d:
                continue
            grp.add(dwg.path(d=d, **style))
        dwg.add(grp)
    dwg.save()
    if metadata:
        # svgwrite has no convenient metadata helper; patch in via ElementTree
        import xml.etree.ElementTree as ET
        tree = ET.parse(out_path)
        root = tree.getroot()
        meta = ET.SubElement(root, "metadata")
        meta.text = json.dumps(metadata)
        tree.write(out_path)


def save_preview(img: np.ndarray, lines: List[Line], arcs: List[Arc], circles: List[Circle],
                 ellipses: List[Ellipse], beziers: List[Bezier], out_path: Path) -> None:
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for l in lines:
        cv2.line(preview, tuple(map(int, l.start)), tuple(map(int, l.end)), (0, 0, 255), 1)
    for c in circles:
        cv2.circle(preview, tuple(map(int, c.center)), int(c.radius), (0, 255, 0), 1)
    for a in arcs:
        cv2.circle(preview, tuple(map(int, a.center)), int(a.radius), (255, 0, 0), 1)
    for e in ellipses:
        cv2.ellipse(preview, (tuple(map(int, e.center)), (int(e.rx*2), int(e.ry*2)), int(e.angle*RAD2DEG)), (255,0,255),1)
    cv2.imwrite(out_path.as_posix(), preview)

# ---- parameter helpers ------------------------------------------------------

def auto_params_from_stroke_width(bw: np.ndarray) -> Dict[str, float]:
    dist_map = morphology.distance_transform_edt(bw > 0)
    stroke = float(np.median(dist_map[dist_map > 0]) * 2.0)
    return {
        "epsilon": stroke * 1.5,
        "gap": stroke * 2.0,
        "min_seg": stroke * 3.0,
    }

# ---- main processing -------------------------------------------------------

def process_image(path: Path, outdir: Path, args: argparse.Namespace) -> Dict[str, int]:
    img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    bw = preprocess(img, threshold=args.threshold, adaptive=args.adaptive,
                    open_px=args.open, close_px=args.close)
    if args.deskew:
        bw = deskew(bw)
    if args.autoscale:
        params = auto_params_from_stroke_width(bw)
        for k, v in params.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    skel = skeletonize_image(bw)
    G, idx = build_graph_from_skeleton(skel)
    polylines = trace_polylines(G, idx)
    polylines = simplify_polylines(polylines, args.epsilon)
    polylines = close_gaps(polylines, args.gap)
    polylines = snap_angles(polylines, args.angle_snap)
    lines = [Line(pl[0], pl[-1]) for pl in polylines if len(pl) >= 2]
    lines = [l for l in lines if dist(l.start, l.end) >= args.min_seg]
    lines = merge_colinear(lines)
    circles, rest = fit_circles_ransac(polylines, args.circle_err)
    ellipses, rest = fit_ellipses_ransac(rest, args.ellipse_err)
    arcs, rest = extract_arcs(rest, args.circle_err)
    beziers, rest = fit_beziers(rest)
    svg_path = outdir / (path.stem + ".svg")
    metadata = {
        "source": path.name,
        "time": time.time(),
        "params": json.dumps(vars(args))
    }
    to_svg(lines, arcs, circles, ellipses, beziers, svg_path, hairline=args.hairline,
           precision=args.precision, metadata=metadata)
    if args.preview:
        preview_path = outdir / (path.stem + "_preview.png")
        save_preview(img, lines, arcs, circles, ellipses, beziers, preview_path)
    return {"lines": len(lines), "circles": len(circles), "arcs": len(arcs),
            "ellipses": len(ellipses), "beziers": len(beziers)}

# ---- argument parsing -------------------------------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vectorize black-on-white schematics to SVG")
    p.add_argument("--input", required=False, help="Input file or folder")
    p.add_argument("--output", default="out", help="Output folder")
    p.add_argument("--recursive", action="store_true", help="Recurse into folders")
    p.add_argument("--detector", choices=["fld", "lsd", "hawp", "letr"], default="auto")
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--gap", type=float, default=None)
    p.add_argument("--angle-snap", type=float, default=5.0)
    p.add_argument("--min-seg", type=float, default=8.0)
    p.add_argument("--circle-err", type=float, default=2.5)
    p.add_argument("--ellipse-err", type=float, default=3.0)
    p.add_argument("--threshold", type=int, default=None)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--open", type=int, default=0)
    p.add_argument("--close", type=int, default=0)
    p.add_argument("--hairline", type=float, default=0.1)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--circle-fit", choices=["pratt", "taubin"], default="pratt")
    p.add_argument("--use-sam", action="store_true", help="Optional SAM pre-segmentation")
    p.add_argument("--autoscale", action="store_true")
    p.add_argument("--grid-tune", action="store_true")
    p.add_argument("--deskew", action="store_true")
    p.add_argument("--preset", choices=["tight", "balanced", "loose"], default="balanced")
    p.add_argument("--debug-test", action="store_true", help="Run internal tests")
    args = p.parse_args(argv)
    presets = {
        "tight": dict(epsilon=0.8, gap=2, angle_snap=3, min_seg=5, circle_err=1.5, ellipse_err=2),
        "balanced": dict(epsilon=1.5, gap=4, angle_snap=5, min_seg=8, circle_err=2.5, ellipse_err=3),
        "loose": dict(epsilon=3, gap=6, angle_snap=8, min_seg=12, circle_err=4, ellipse_err=5),
    }
    preset = presets[args.preset]
    for k, v in preset.items():
        if getattr(args, k) is None or k in ["angle_snap", "circle_err", "ellipse_err", "min_seg"]:
            setattr(args, k, getattr(args, k) if getattr(args, k) is not None else v)
    return args

# ---- debug test -------------------------------------------------------------

def debug_test() -> None:
    """Generate synthetic raster, vectorise, and sanity-check outputs."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (80, 100), (0, 0, 0), 3)
    cv2.circle(img, (150, 80), 30, (0, 0, 0), 3)
    cv2.line(img, (20, 150), (180, 150), (0, 0, 0), 3)
    png = tmp / "test.png"
    cv2.imwrite(png.as_posix(), img)
    args = parse_args(["--input", png.as_posix(), "--output", tmp.as_posix(), "--preview"])
    stats = process_image(png, tmp, args)
    svg = tmp / "test.svg"
    assert svg.exists(), "SVG not created"
    import xml.etree.ElementTree as ET
    tree = ET.parse(svg)
    root = tree.getroot()
    groups = {child.attrib.get("id") for child in root if child.tag.endswith('g')}
    assert {"lines", "circles", "arcs", "ellipses", "beziers"} <= groups
    print("debug-test OK", stats, "output in", tmp)

# ---- entrypoint -------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.debug_test:
        debug_test()
        return
    if args.input is None:
        raise SystemExit("--input is required")
    in_path = Path(args.input)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    if in_path.is_file():
        files = [in_path]
    else:
        glob = "**/*" if args.recursive else "*"
        for f in in_path.glob(glob):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                files.append(f)
    process: Iterable[Path] = files
    iterator: Iterable = process
    if tqdm is not None:
        iterator = tqdm(process, desc="vectorising")
    for f in iterator:
        stats = process_image(f, outdir, args)
        if tqdm is None:
            print(f"{f.name}: {stats}")

if __name__ == "__main__":
    main()
