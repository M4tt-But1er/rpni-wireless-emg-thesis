"""Scientific-style schematic of RPNI formation and function (clean vector style), four panels (17 Nov 2026):
a severed nerve fascicle and a separate denervated free muscle graft; b fascicle implanted, graft wrapped around it and
sutured closed along a horizontal seam; c revascularisation and reinnervation;
d attempted movement: a small nerve signal (uV) produces a large graft EMG (mV) at the electrode (biological amplifier).
Outputs rpni_formation_scientific[_nolabels].png (300 dpi) and .pdf. Usage: python make_rpni_figure_scientific.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch

FONT = "Arial"
INK, GREY, OUTLINE = "#262626", "#6b6b6b", "#8a8a8a"
NERVE, NERVE_D, AXON = "#e7b04b", "#a8741c", "#d9962a"
MUSCLE, MUSCLE_D, MUSCLE_PALE = "#c8574b", "#8e2f26", "#e8b9b1"
ART, VEIN, CAP = "#b3261e", "#2f55a4", "#5e0b08"
SKIN, SKIN_D = "#f2dcc7", "#c9a88a"
ELECTRODE = "#a9adb4"
LW = 1.0
FS = 1.0                                                      # font scale (set per output in draw)
plt.rcParams.update({"font.family": FONT, "font.size": 8})


def bez(ax, pts, color, lw=LW, clip=None, z=5, alpha=1.0, ls="-"):
    P = np.asarray(pts, float)
    t = np.linspace(0, 1, 100)[:, None]
    if len(P) == 2:
        xy = (1 - t) * P[0] + t * P[1]
    elif len(P) == 3:
        xy = (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]
    else:
        xy = (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
    ln, = ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, ls=ls, solid_capstyle="round", zorder=z)
    if clip is not None:
        ln.set_clip_path(clip)
    return ln


def inset(ax, c, r):
    ax.add_patch(Circle(c, r, fc="white", ec=OUTLINE, lw=1.1, zorder=1))
    clip = Circle(c, r - 0.4, transform=ax.transData, visible=False)
    ax.add_patch(clip)
    return clip


def nerve(ax, x0, x1, y, clip, width=4.6):
    xs = np.linspace(x0, x1, 40)
    ys = y + 0.5 * np.sin((xs - x0) / 8.0)
    p = Polygon(np.r_[np.c_[xs, ys + width / 2], np.c_[xs[::-1], ys[::-1] - width / 2]], fc=NERVE, ec=NERVE_D, lw=LW, zorder=3)
    ax.add_patch(p)
    p.set_clip_path(clip)
    for f in (-1, 0, 1):                                       # fascicle lines
        bez(ax, [(x0, y + f * 1.2), ((x0 + x1) / 2, y + f * 1.2 + 0.3), (x1, y + f * 1.2)], NERVE_D, 0.4, clip=clip, z=4, alpha=0.6)
    return p


def graft(ax, x, y, w, h, fill, clip, depth=None, alpha=1.0, p=2.6):
    """Rounded muscle belly (superellipse) in the box (x, y, w, h): soft top highlight and lower shade, fibres that follow the
    outline and taper towards both ends."""
    from matplotlib.patches import Ellipse
    xc, yc, a, b = x + w / 2, y + h / 2, w / 2, h / 2
    th = np.linspace(0, 2 * np.pi, 240)
    ct, st = np.cos(th), np.sin(th)
    pts = np.c_[xc + a * np.sign(ct) * np.abs(ct) ** (2 / p), yc + b * np.sign(st) * np.abs(st) ** (2 / p)]
    body = Polygon(pts, fc=fill, ec=MUSCLE_D, lw=LW, alpha=alpha, zorder=5)
    ax.add_patch(body)
    body.set_clip_path(clip)
    for (ey, eh, col, al) in ((yc + 0.42 * b, 0.9 * b, "white", 0.22), (yc - 0.75 * b, 0.9 * b, MUSCLE_D, 0.22)):
        e = Ellipse((xc, ey), 1.9 * a, eh, fc=col, ec="none", alpha=al * alpha, zorder=5)
        ax.add_patch(e)
        e.set_clip_path(body)
    u = np.linspace(-1, 1, 120)
    prof = (1 - np.abs(u) ** p) ** (1 / p)
    for f in np.linspace(-0.82, 0.82, 8):                      # fibres converge towards the ends
        ln, = ax.plot(xc + a * 0.97 * u, yc + f * b * prof, color=MUSCLE_D, lw=0.45, alpha=0.55, zorder=6)
        ln.set_clip_path(body)
    return body


def trace(ax, x0, y0, w, amp, kind, color, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 600)
    if kind == "emg":
        s = rng.standard_normal(t.size) * np.exp(-((t - 0.5) / 0.17) ** 2) + 0.04 * rng.standard_normal(t.size)
    else:
        s = 0.05 * rng.standard_normal(t.size)
        for c in (0.2, 0.36, 0.5, 0.64, 0.8):
            s += np.exp(-((t - c) / 0.005) ** 2) - 0.45 * np.exp(-((t - c - 0.011) / 0.007) ** 2)
    s = s / np.abs(s).max()
    ax.plot(x0 + w * t, y0 + amp * s, color=color, lw=0.7, zorder=8)


def label(ax, text, xy_text, xy_pt, ha="left", va="center"):
    ax.annotate(text, xy=xy_pt, xytext=xy_text, ha=ha, va=va, fontsize=7 * FS, color=INK, zorder=10,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=1, shrinkB=0.5))


def seq_arrow(ax, p0, p1, text, labels):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=4,head_width=2.2", color=GREY, lw=1.2, zorder=2))
    if labels:
        ax.text((p0[0] + p1[0]) / 2, p0[1] + 18, text, ha="center", va="bottom", fontsize=6.3 * FS, color=GREY, linespacing=1.15)


def ring_check(fig, ax, centres, R, margin=1.2):
    """Report every text whose box touches or crosses a ring (inside and outside at once, or within `margin` of the ring)."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        xs, ys = np.linspace(x0, x1, 25), np.linspace(y0, y1, 9)
        pts = np.r_[np.c_[xs, np.full_like(xs, y0)], np.c_[xs, np.full_like(xs, y1)], np.c_[np.full_like(ys, x0), ys], np.c_[np.full_like(ys, x1), ys]]
        for i, c in enumerate(centres):
            d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
            if (d.min() < R + margin) and (d.max() > R - margin):
                bad.append((t.get_text().replace(chr(10), " "), "abcd"[i], round(d.min() - R, 1), round(d.max() - R, 1)))
    boxes = []
    for t in ax.texts:
        if t.get_text().strip():
            bb = t.get_window_extent()
            boxes.append((t.get_text().replace(chr(10), " "), inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (a0, a1), (b0, b1) = boxes[i][1], boxes[j][1]
            if a0[0] < b1[0] and b0[0] < a1[0] and a0[1] < b1[1] and b0[1] < a1[1]:
                bad.append(("overlap", boxes[i][0], boxes[j][0]))
    return bad


def draw(labels=True, out="rpni_formation_scientific", width=13.98, fs=1.0):
    global FS
    FS = fs
    from matplotlib.patches import Ellipse
    X0, X1, Y0, Y1 = 66, 434, 4, 128
    fig, ax = plt.subplots(figsize=(width, width * (Y1 - Y0) / (X1 - X0)))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal")
    ax.axis("off")

    R, cy = 37, 66
    C = [(108, cy), (197, cy), (286, cy), (375, cy)]
    GX, GY, GW, GH = -6, -12, 30, 24

    # ---------------- a: severed nerve fascicle and a separate, denervated free muscle graft
    c = C[0]
    clip = inset(ax, c, R)
    xe = c[0] - 9                                               # cut end of the fascicle
    nerve(ax, c[0] - R, xe, c[1], clip)
    ye = c[1] + 0.5 * np.sin((xe - (c[0] - R)) / 8.0)
    ax.add_patch(Ellipse((xe, ye), 2.6, 4.6, fc=NERVE, ec=NERVE_D, lw=LW, zorder=5))
    for dx, dy in ((-0.25, 1.1), (0.25, -0.1), (-0.15, -1.3)):  # fascicles seen in the cut face
        ax.add_patch(Circle((xe + dx, ye + dy), 0.45, fc=NERVE_D, ec="none", alpha=0.6, zorder=6))
    graft(ax, c[0] + 1, c[1] + GY, GW - 3, GH, MUSCLE_PALE, clip, alpha=0.82)
    if labels:
        label(ax, "Severed nerve\nfascicle", (c[0] - 28, c[1] + 15), (c[0] - 16, c[1] + 2.4))
        label(ax, "Free muscle graft\n(denervated)", (c[0] + 7, c[1] - 17), (c[0] + 11, c[1] - 11.8), ha="center", va="top")

    # ---------------- b: fascicle implanted, graft wrapped around it and sutured closed along a horizontal seam
    c = C[1]
    clip = inset(ax, c, R)
    nerve(ax, c[0] - R, c[0] + 6, c[1], clip)
    graft(ax, c[0] + GX, c[1] + GY, GW, GH, MUSCLE_PALE, clip, alpha=0.82)
    sx0, sx1, sy = c[0] + GX + 2.5, c[0] + GX + GW - 4, c[1] + 4.5
    bez(ax, [(sx0, sy), ((sx0 + sx1) / 2, sy + 0.9), (sx1, sy)], MUSCLE_D, 0.9, z=8)      # seam where the graft edges meet
    for x in np.linspace(sx0 + 1.5, sx1 - 1.5, 7):                                         # sutures across the seam
        yy = sy + 0.9 * 4 * ((x - sx0) / (sx1 - sx0)) * (1 - (x - sx0) / (sx1 - sx0))
        bez(ax, [(x - 0.7, yy - 1.5), (x + 0.7, yy + 1.5)], INK, 0.85, z=9)
    if labels:
        label(ax, "Sutures", (c[0] - 26, c[1] + 15), (sx0 + 3, sy + 1.2))
        label(ax, "Graft wrapped around\nthe nerve end", (c[0] + 5, c[1] - 17), (c[0] + 12, c[1] - 11.8), ha="center", va="top")

    # ---------------- c: revascularisation and reinnervation
    c = C[2]
    clip = inset(ax, c, R)
    nerve(ax, c[0] - R, c[0] + 6, c[1], clip)
    g = graft(ax, c[0] + GX, c[1] + GY, GW, GH, MUSCLE, clip)
    root = (c[0] + 3, c[1])
    for k in range(6):                                         # motor axons and end plates
        ty = c[1] - 8.5 + k * 17 / 5
        tx = c[0] + (8 if k % 2 else 11)
        bez(ax, [root, (c[0] + 6, (c[1] + ty) / 2 + (ty - c[1]) * 0.1), (tx, ty)], AXON, 0.9, clip=g, z=7)
        ax.plot(tx, ty, "o", ms=2.4, color=NERVE, mec=NERVE_D, mew=0.5, zorder=8)
    rng = np.random.default_rng(11)                           # capillary bed: fine vessels between the fibres, with cross-links
    x0, x1 = c[0] + 13, c[0] + GX + GW - 0.8
    rows = [c[1] + GY + (k + 0.5) * GH / 8 for k in range(8)]
    for y in rows:
        xs = np.linspace(x0, x1, 40)
        ys = y + 0.35 * np.sin((xs - x0) / 1.6 + rng.uniform(0, 6))
        ln, = ax.plot(xs, ys, color=CAP, lw=0.65, alpha=0.95, zorder=7)
        ln.set_clip_path(g)
    for y_lo, y_hi in zip(rows[:-1], rows[1:]):
        for xc in rng.uniform(x0 + 1, x1 - 1, 2):
            bez(ax, [(xc, y_lo), (xc + rng.uniform(-0.8, 0.8), (y_lo + y_hi) / 2), (xc + rng.uniform(-0.6, 0.6), y_hi)], CAP, 0.65, clip=g, z=7, alpha=0.95)
    if labels:
        label(ax, "Motor axons and\nend plates", (c[0] - 27, c[1] + 16), (c[0] + 6, c[1] + 3.5))
        label(ax, "Capillaries", (c[0] + 7, c[1] - 17), (c[0] + 18, c[1] - 7.5), va="top")
        label(ax, "Reinnervated\ngraft", (c[0] - 28, c[1] - 16), (c[0] - 3, c[1] - 9.5))

    # ---------------- d: attempted movement -> graft EMG
    c = C[3]
    clip = inset(ax, c, R)
    nerve(ax, c[0] - R, c[0] + 6, c[1] - 4, clip)
    g = graft(ax, c[0] + GX, c[1] + GY - 4, GW, GH, MUSCLE, clip)
    for k in range(5):
        ty = c[1] - 12 + k * 4.5
        bez(ax, [(c[0] + 4, c[1] - 4), (c[0] + 9, (c[1] - 4 + ty) / 2), (c[0] + 15, ty)], AXON, 0.9, clip=g, z=7)
    ax.add_patch(FancyArrowPatch((c[0] - 31, c[1] - 10.5), (c[0] - 14, c[1] - 10.5), arrowstyle="-|>,head_length=3,head_width=1.6",
                                 color=NERVE_D, lw=0.9, zorder=8))                      # direction of the motor command
    ax.add_patch(Circle((c[0] + 12, c[1] - 9), 2.0, fc=ELECTRODE, ec=INK, lw=0.8, zorder=9))
    bez(ax, [(c[0] + 12, c[1] - 9), (c[0] + 36, c[1] - 15), (c[0] + 34, c[1] + 17), (c[0] + 22, c[1] + 17)], GREY, 1.0, clip=clip, z=8)
    trace(ax, c[0] - 31, c[1] + 6, 19, 1.1, "nerve", NERVE_D, 3)
    trace(ax, c[0] - 3, c[1] + 17, 24, 4.5, "emg", MUSCLE_D, 5)
    if labels:
        ax.text(c[0] - 17.0, c[1] + 9.2, "Nerve signal (µV)", ha="center", va="bottom", fontsize=6.5 * FS, color=NERVE_D)
        ax.text(c[0] + 9, c[1] + 22.5, "Graft EMG (mV)", ha="center", va="bottom", fontsize=6.5 * FS, color=MUSCLE_D)
        ax.text(c[0] - 17.5, c[1] - 12.5, "Motor command", ha="center", va="top", fontsize=6.5 * FS, color=NERVE_D)
        label(ax, "Electrode", (c[0] + 2, c[1] - 24), (c[0] + 11.2, c[1] - 10.8))

    # ---------------- sequence arrows, panel letters, captions
    for i, txt in enumerate(["Surgery", "Weeks to\nmonths", "Attempted\nmovement"]):
        seq_arrow(ax, (C[i][0] + R + 2, cy), (C[i + 1][0] - R - 2, cy), txt, labels)
    for letter, (x, _) in zip("abcd", C):
        ax.text(x - R, cy + R + 2, letter, fontsize=11 * FS, fontweight="bold", color=INK, ha="left", va="top")
    if labels:
        caps = ["Severed nerve fascicle and a\nseparate, denervated muscle graft",
                "Fascicle implanted and the graft\nsutured closed around it",
                "Revascularisation and reinnervation\nof the graft by motor axons",
                "Motor command activates the graft:\nsmall nerve signal, large recordable EMG"]
        for (x, _), t in zip(C, caps):
            ax.text(x, cy - R - 4, t, ha="center", va="top", fontsize=7.5 * FS, color=INK, linespacing=1.3)
    if labels:
        for b in ring_check(fig, ax, C, R):
            print(f"  {out}:", b)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    from PIL import Image, ImageChops
    im = Image.open(f"{out}.png").convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    im.crop((max(box[0] - 30, 0), max(box[1] - 30, 0), min(box[2] + 30, im.width), min(box[3] + 30, im.height))).save(f"{out}.png", optimize=True)


if __name__ == "__main__":
    draw(True, "rpni_formation_scientific")
    draw(False, "rpni_formation_scientific_nolabels")
    draw(True, "rpni_formation_thesis", width=9.5, fs=0.85)          # same panel scale as the three-panel thesis version
    print("done")
