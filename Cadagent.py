#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potrace_vector_tests.py
Vectorización avanzada con Potrace + refit geométrico (líneas/arcos/círculos/elipses).
- Preprocesa (cierra gaps) -> Potrace (CLI o pure-python fallback) -> parsea SVG -> 
  muestrea puntos -> RDP -> fit line/circle/ellipse -> limpia -> JSON + SVG + overlays.
"""

import argparse, json, math, os, shutil, subprocess, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

# parsing SVG (de potrace) y escritura final
from svgpathtools import svg2paths, Path, Line, CubicBezier, QuadraticBezier, Arc, wsvg
from shapely.geometry import LineString, Point

# ==== utilidades numéricas =====================================================
def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker; points: (N,2). Devuelve índice booleano de puntos conservados."""
    if len(points) < 3:
        return np.ones(len(points), dtype=bool)
    start, end = points[0], points[-1]
    line_vec = end - start
    if np.allclose(line_vec, 0):
        d = np.linalg.norm(points - start, axis=1)
    else:
        u = line_vec / (np.linalg.norm(line_vec) + 1e-12)
        proj = np.dot(points - start, u)
        nearest = start + np.outer(proj, u)
        d = np.linalg.norm(points - nearest, axis=1)
    idx = int(np.argmax(d))
    dmax = d[idx]
    if dmax > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        out = np.concatenate([left[:-1], right])
        return out
    else:
        out = np.zeros(len(points), dtype=bool)
        out[0] = True
        out[-1] = True
        return out

def fit_circle_taubin(pts: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Ajuste de círculo (Taubin). Devuelve centro (cx,cy), radio r, error RMS."""
    x = pts[:, 0]; y = pts[:, 1]
    x_m = x.mean(); y_m = y.mean()
    u = x - x_m; v = y - y_m
    Suu = np.dot(u, u); Svv = np.dot(v, v); Suv = np.dot(u, v)
    Suuu = np.dot(u, u*u); Svvv = np.dot(v, v*v)
    Suvv = np.dot(u, v*v); Svuu = np.dot(v, u*u)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc, vc = 0.0, 0.0
    cx = x_m + uc; cy = y_m + vc
    r = np.sqrt(uc*uc + vc*vc + (Suu + Svv)/len(pts))
    err = np.sqrt(((np.hypot(x - cx, y - cy) - r) ** 2).mean())
    return np.array([cx, cy]), r, err

def fit_ellipse_direct(pts: np.ndarray):
    """Ajuste de elipse (Fitzgibbon). Devuelve (cx,cy, rx,ry, rot), err."""
    x = pts[:, 0][:, None]; y = pts[:, 1][:, None]
    D = np.hstack([x*x, x*y, y*y, x, y, np.ones_like(x)])
    S = np.dot(D.T, D)
    C = np.zeros([6, 6]); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    try:
        eigval, eigvec = np.linalg.eig(np.linalg.inv(S).dot(C))
    except np.linalg.LinAlgError:
        return None, np.inf
    a = eigvec[:, np.argmax(eigval.real)]
    # Parametrización a elipse
    b,c,d,f,g,a0 = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b - a0*c
    if num == 0: return None, np.inf
    x0 = (c*d - b*f) / num
    y0 = (a0*f - b*d) / num
    up = 2 * (a0*f*f + c*d*d + g*b*b - 2*b*d*f - a0*c*g)
    down1 = (b*b - a0*c) * ((c - a0) + np.sqrt((a0 - c) ** 2 + 4*b*b))
    down2 = (b*b - a0*c) * ((c - a0) - np.sqrt((a0 - c) ** 2 + 4*b*b))
    if down1 == 0 or down2 == 0: return None, np.inf
    rx = np.sqrt(np.abs(up / down1))
    ry = np.sqrt(np.abs(up / down2))
    theta = 0.5 * math.atan2(2*b, a0 - c)
    cx, cy = float(x0), float(y0)
    err = np.mean(np.abs(((pts[:,0]-cx)*np.cos(theta)+(pts[:,1]-cy)*np.sin(theta))**2/rx**2 +
                         ((pts[:,0]-cx)*np.sin(theta)-(pts[:,1]-cy)*np.cos(theta))**2/ry**2 - 1))
    rx, ry = float(rx), float(ry)
    # normaliza para rx>=ry
    if ry > rx:
        rx, ry = ry, rx
        theta += math.pi/2
    return (cx, cy, rx, ry, float(theta % math.pi)), float(err)

# ==== preprocesado y ejecución de potrace ======================================
def binarize_and_close(img_path: str, outdir: str, close_px: int = 2) -> str:
    os.makedirs(outdir, exist_ok=True)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert gray is not None, f"No se pudo leer {img_path}"
    # umbral adaptativo estable para dibujos lineales en blanco/negro
    thr1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_px*2+1, close_px*2+1))
    closed = cv2.morphologyEx(thr1, cv2.MORPH_CLOSE, kernel, iterations=1)
    # PBM para potrace
    pbm_path = os.path.join(outdir, "prep.pbm")
    Image.fromarray((closed>0).astype(np.uint8)*255).convert("1").save(pbm_path)
    cv2.imwrite(os.path.join(outdir,"debug_bin.png"), closed)
    return pbm_path

def have_potrace_cli() -> bool:
    return shutil.which("potrace") is not None

def run_potrace_cli(pbm_path: str, out_svg: str,
                    turdsize=0, alphamax=1.0, opttolerance=0.2, longcurve=True):
    args = [
        "potrace", pbm_path, "--svg", "-o", out_svg,
        "--turdsize", str(turdsize),
        "--alphamax", str(alphamax),
        "--opttolerance", str(opttolerance),
        "--unit", "1",
    ]
    if longcurve: args.append("--longcurve")
    subprocess.check_call(args)

def run_potrace_fallback(pbm_path: str, out_svg: str):
    """Fallback pure-python (tatarize/potrace)."""
    import potrace as pypotr
    bmp = Image.open(pbm_path).convert("1")
    bmp = np.array(bmp).astype(np.uint8)
    bmp = (bmp > 0).astype(np.uint8)
    bmp = np.ascontiguousarray(bmp)
    bmpobj = pypotr.Bitmap(bmp)
    path = bmpobj.trace()
    # export SVG simple
    parts = []
    scale = 1.0
    for curve in path:
        pt = curve.start_point
        d = f"M {pt.x*scale} {pt.y*scale} "
        for segment in curve:
            if segment.is_corner:
                c = segment.c
                d += f"L {c.x*scale} {c.y*scale} "
            else:
                a,b,c = segment.c1, segment.c2, segment.end_point
                d += f"C {a.x*scale} {a.y*scale} {b.x*scale} {b.y*scale} {c.x*scale} {c.y*scale} "
        d += "Z"
        parts.append(d)
    svg = '<svg xmlns="http://www.w3.org/2000/svg">\n' + \
          "".join(f'<path d="{d}" fill="none" stroke="black" stroke-width="1"/>\n' for d in parts) + \
          '</svg>'
    with open(out_svg,"w",encoding="utf-8") as f: f.write(svg)

# ==== muestreo de paths SVG y clasificación geométrica =========================
def sample_svg(svg_path: str, step: float = 2.0) -> List[np.ndarray]:
    """Devuelve lista de polilíneas (Nx2) muestreadas a lo largo de cada path."""
    paths, _ = svg2paths(svg_path)
    polylines = []
    for p in paths:
        length = p.length(error=1e-4)
        n = max(2, int(length/step))
        ts = np.linspace(0,1,n)
        pts = np.array([ [p.point(t).real, p.point(t).imag] for t in ts ], dtype=np.float64)
        polylines.append(pts)
    return polylines

@dataclass
class LineSeg:  x1:int; y1:int; x2:int; y2:int
@dataclass
class ArcSeg:   cx:float; cy:float; r:float; a0:float; a1:float  # ángulos grados
@dataclass
class Circle:   x:float; y:float; r:float
@dataclass
class Ellipse:  x:float; y:float; rx:float; ry:float; rot:float

def split_poly_by_curvature(pts: np.ndarray, win=5, k_thresh=0.01) -> List[np.ndarray]:
    """Divide una polilínea en trozos casi lineales/curvos usando curvatura discreta."""
    if len(pts) <= 2: return [pts]
    k = []
    for i in range(1, len(pts)-1):
        v1 = pts[i] - pts[i-1]
        v2 = pts[i+1] - pts[i]
        a1 = np.linalg.norm(v1)+1e-12
        a2 = np.linalg.norm(v2)+1e-12
        cosang = np.dot(v1,v2)/(a1*a2)
        ang = np.arccos(np.clip(cosang,-1,1))
        k.append(ang)
    k = np.array([0,*k,0])
    splits = [0]
    for i in range(1,len(k)-1):
        if k[i] > k_thresh and (i - splits[-1]) > win:
            splits.append(i)
    splits.append(len(pts)-1)
    chunks = [pts[s:e+1] for s,e in zip(splits[:-1], splits[1:])]
    return chunks

def classify_chunk(pts: np.ndarray, line_eps=0.8, circle_rms=0.8):
    """Devuelve ('line', LineSeg) | ('arc', ArcSeg) | ('ellipse', Ellipse) | None."""
    if len(pts) < 2: return None
    # 1) ¿línea?
    mask = rdp(pts, line_eps)
    pts2 = pts[mask]
    if len(pts2) == 2:
        (x1,y1),(x2,y2) = pts2[0], pts2[-1]
        return ("line", LineSeg(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    # 2) ¿círculo/ arco?
    c,(r),err = fit_circle_taubin(pts)
    if err < circle_rms and r>2:
        # definimos arco con ángulos a partir del centro
        angs = np.degrees(np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0]))
        a0, a1 = float(angs[0]), float(angs[-1])
        # normaliza para que respete sentido
        return ("arc", ArcSeg(float(c[0]), float(c[1]), float(r), a0, a1))
    # 3) ¿elipse?
    ell, e_err = fit_ellipse_direct(pts)
    if ell and e_err < 0.08:
        cx,cy,rx,ry,rot = ell
        if rx>3 and ry>2:
            # círculo especial
            if rx/ry < 1.15:
                return ("circle", Circle(float(cx),float(cy), float((rx+ry)/2)))
            return ("ellipse", Ellipse(float(cx),float(cy), float(rx),float(ry), float(rot*180/math.pi)))
    return None

def merge_collinear(lines: List[LineSeg], angle_tol=3.0, dist_tol=3.0) -> List[LineSeg]:
    """Une segmentos colineales contiguos."""
    def angle(l:LineSeg):
        return math.degrees(math.atan2(l.y2-l.y1, l.x2-l.x1))
    used=[False]*len(lines); out=[]
    for i,l in enumerate(lines):
        if used[i]: continue
        ax=angle(l)
        x1,y1,x2,y2 = l.x1,l.y1,l.x2,l.y2
        changed=True
        used[i]=True
        while changed:
            changed=False
            for j,m in enumerate(lines):
                if used[j]: continue
                if abs(angle(m)-ax) < angle_tol:
                    # comprobamos proximidad a extremos
                    if np.hypot(x2-m.x1,y2-m.y1) < dist_tol:
                        x2,y2 = m.x2,m.y2; used[j]=True; changed=True
                    elif np.hypot(m.x2-x1,m.y2-y1) < dist_tol:
                        x1,y1 = m.x1,m.y1; used[j]=True; changed=True
        out.append(LineSeg(x1,y1,x2,y2))
    return out

def snap_endpoints(lines: List[LineSeg], tol=2.0) -> List[LineSeg]:
    """Snapea nodos finales cercanos a su media (cierra pequeñas holguras)."""
    pts=[]
    for l in lines: pts += [(l.x1,l.y1),(l.x2,l.y2)]
    pts=np.array(pts,dtype=float)
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        close = np.where((d>0) & (d<tol))[0]
        if len(close)>0:
            mean = pts[[i,*close]].mean(axis=0)
            pts[i]=mean; pts[close]=mean
    out=[]; it=iter(pts)
    for l in lines:
        p1=next(it); p2=next(it)
        out.append(LineSeg(int(round(p1[0])),int(round(p1[1])),
                           int(round(p2[0])),int(round(p2[1]))))
    return out

# ==== pipeline =================================================================
def vectorize_with_potrace(input_image: str, outdir: str,
                           step=2.0, rdp_eps=0.8,
                           circle_rms=0.8, line_eps=0.8,
                           close_px=2):
    os.makedirs(outdir, exist_ok=True)
    pbm_path = binarize_and_close(input_image, outdir, close_px=close_px)

    raw_svg = os.path.join(outdir, "trace_raw.svg")
    if have_potrace_cli():
        # Ajusta alphamax/opttolerance para suavidad vs fidelidad
        run_potrace_cli(pbm_path, raw_svg, turdsize=0, alphamax=1.0, opttolerance=0.2, longcurve=True)
    else:
        print("[WARN] potrace CLI no encontrado; usando fallback puro-python (más lento).")
        run_potrace_fallback(pbm_path, raw_svg)

    # muestreo de paths trazados
    polys = sample_svg(raw_svg, step=step)
    # visualización de puntos
    debug_points = np.ones((1024, 1024, 3), dtype=np.uint8)*255
    for poly in polys:
        for p in poly.astype(int):
            cv2.circle(debug_points, tuple(p), 1, (0,0,0), -1)
    cv2.imwrite(os.path.join(outdir,"trace_points.png"), debug_points)

    # clasificación por trozos
    lines: List[LineSeg] = []
    arcs:  List[ArcSeg] = []
    circles: List[Circle] = []
    ellipses: List[Ellipse] = []

    for poly in polys:
        chunks = split_poly_by_curvature(poly, win=4, k_thresh=0.12)
        for ch in chunks:
            # simplifica cada trozo y clasifícalo
            keep = rdp(ch, rdp_eps)
            ch2 = ch[keep]
            cls = classify_chunk(ch2, line_eps=line_eps, circle_rms=circle_rms)
            if cls is None: 
                # si no clasifica, intenta línea por mínimos cuadrados
                (vx,vy,x0,y0) = cv2.fitLine(ch2.astype(np.float32), cv2.DIST_L2,0,0.01,0.01)
                p1 = ch2[0]; p2 = ch2[-1]
                lines.append(LineSeg(int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1])))
                continue
            typ,obj = cls
            if   typ=="line":    lines.append(obj)
            elif typ=="arc":     arcs.append(obj)
            elif typ=="circle":  circles.append(obj)
            elif typ=="ellipse": ellipses.append(obj)

    # limpieza
    lines = merge_collinear(lines, angle_tol=3.0, dist_tol=3.0)
    lines = snap_endpoints(lines, tol=2.0)

    # export JSON
    data = {
        "lines":[vars(l) for l in lines],
        "arcs":[vars(a) for a in arcs],
        "circles":[vars(c) for c in circles],
        "ellipses":[vars(e) for e in ellipses],
        "_meta":{"note":"potrace+refit","params":{
            "step":step,"rdp_eps":rdp_eps,"circle_rms":circle_rms,"line_eps":line_eps,"close_px":close_px
        }}
    }
    with open(os.path.join(outdir, "vector.json"),"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2)

    # export SVG bonito
    H,W = 1536,1024  # si los conoces; si no, recógelo de la imagen
    svg_elems = []
    for l in lines:
        svg_elems.append(f'<line x1="{l.x1}" y1="{l.y1}" x2="{l.x2}" y2="{l.y2}" stroke="red" stroke-width="1"/>')
    for a in arcs:
        # convierte a path con parámetros de arco
        large = 1 if abs(a.a1-a.a0)>180 else 0
        # punto inicial y final
        a0 = math.radians(a.a0); a1 = math.radians(a.a1)
        x0 = a.cx + a.r*math.cos(a0); y0 = a.cy + a.r*math.sin(a0)
        x1 = a.cx + a.r*math.cos(a1); y1 = a.cy + a.r*math.sin(a1)
        svg_elems.append(
            f'<path d="M {x0} {y0} A {a.r} {a.r} 0 {large} 1 {x1} {y1}" fill="none" stroke="blue" stroke-width="1"/>'
        )
    for c in circles:
        svg_elems.append(f'<circle cx="{c.x}" cy="{c.y}" r="{c.r}" fill="none" stroke="cyan" stroke-width="1"/>')
    for e in ellipses:
        svg_elems.append(
            f'<ellipse cx="{e.x}" cy="{e.y}" rx="{e.rx}" ry="{e.ry}" transform="rotate({e.rot} {e.x} {e.y})" '
            f'fill="none" stroke="magenta" stroke-width="1"/>'
        )
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n' + \
          "\n".join(svg_elems) + "\n</svg>"
    with open(os.path.join(outdir, "vector.svg"),"w",encoding="utf-8") as f: f.write(svg)

    # overlay de depuración
    img = cv2.imread(input_image, cv2.IMREAD_COLOR)
    if img is not None:
        overlay = img.copy()
        for l in lines: cv2.line(overlay,(l.x1,l.y1),(l.x2,l.y2),(0,0,255),2)
        for c in circles: cv2.circle(overlay,(int(c.x),int(c.y)),int(c.r),(255,255,0),2)
        for e in ellipses: cv2.ellipse(overlay,(int(e.x),int(e.y)),(int(e.rx),int(e.ry)),
                                       e.rot,0,360,(255,0,255),2)
        cv2.imwrite(os.path.join(outdir,"overlay.png"), overlay)

    print(f"[OK] Salidas en: {outdir}")
    return data

# ==== CLI ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="ruta a input.png")
    ap.add_argument("--outdir", default="potrace_out", help="carpeta de salida")
    ap.add_argument("--step", type=float, default=2.0, help="paso de muestreo sobre paths de potrace (px)")
    ap.add_argument("--rdp", type=float, default=0.8, help="epsilon RDP por trozo")
    ap.add_argument("--circle_rms", type=float, default=0.8, help="umbral RMS para círculo")
    ap.add_argument("--line_eps", type=float, default=0.8, help="epsilon línea (RDP)")
    ap.add_argument("--close_px", type=int, default=2, help="radio del cierre morfológico previo")
    args = ap.parse_args()
    vectorize_with_potrace(args.inp, args.outdir, step=args.step, rdp_eps=args.rdp,
                           circle_rms=args.circle_rms, line_eps=args.line_eps,
                           close_px=args.close_px)

if __name__ == "__main__":
    main()
