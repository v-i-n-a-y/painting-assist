# Copyright 2026 Vinay Williams

"""Programmatic app icon generator for Painting Assist.

Draws a rounded-square canvas with a coarse-to-fine blurred colour-block
motif (a few flat colour masses fading into a soft blur) plus a subtle grid
hint, echoing the app's crop / blur / colour-group / grid tools.

Run with:  uv run python painting_assist/resources/make_icon.py
"""

from __future__ import annotations

import os
import subprocess

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 1024
SS = 4  # supersampling factor for crisp edges


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    grad = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        grad.putpixel(
            (0, y),
            tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)),
        )
    return grad.resize((size, size))


def build_base(size: int) -> Image.Image:
    """The full icon at the given size, drawn at supersampled resolution."""
    S = size * SS
    radius = int(S * 0.225)

    # --- Background: soft warm-to-cool gradient, like a toned canvas ---
    img = vertical_gradient(S, (245, 241, 233), (222, 226, 233)).convert("RGBA")

    # --- Coarse colour masses: a painter's simplified block study ---
    # A warm/cool landscape-ish abstraction inset within the canvas.
    inset = int(S * 0.14)
    box = (inset, inset, S - inset, S - inset)
    bw = box[2] - box[0]
    bh = box[3] - box[1]

    blocks = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blocks)

    palette = [
        (232, 168, 74),  # warm ochre
        (214, 96, 77),  # terracotta red
        (108, 142, 155),  # slate teal
        (66, 88, 120),  # deep blue
        (196, 204, 168),  # sage
    ]

    # Diagonal bands of flat colour (coarse "colour group" masses).
    # Draw as polygons sweeping across the inset region.
    xs = [0.0, 0.28, 0.5, 0.72, 1.0]
    skew = 0.22
    for i in range(len(xs) - 1):
        x0 = box[0] + xs[i] * bw
        x1 = box[0] + xs[i + 1] * bw
        col = palette[i]
        pts = [
            (x0, box[1]),
            (x1, box[1]),
            (x1 - skew * bw, box[3]),
            (x0 - skew * bw, box[3]),
        ]
        bd.polygon(pts, fill=col + (255,))

    # A simple sun/focal circle in the warm area for interest.
    r = int(bw * 0.13)
    cx = box[0] + int(bw * 0.30)
    cy = box[1] + int(bh * 0.34)
    bd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(250, 224, 150, 255))

    # Clip the blocks to the inner canvas rounded rect.
    inner_radius = int(S * 0.10)
    inner_mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(inner_mask).rounded_rectangle(box, radius=inner_radius, fill=255)

    # --- Coarse-to-fine blur: sharp on the left, blurred to the right ---
    # Blend a heavily blurred copy using a horizontal ramp so the motif
    # reads as "blur reference" — a signature Painting Assist tool.
    blurred = blocks.filter(ImageFilter.GaussianBlur(int(S * 0.05)))
    ramp = Image.new("L", (S, 1))
    for x in range(S):
        t = x / (S - 1)
        # ease so blur ramps up toward the right third
        a = max(0.0, (t - 0.45) / 0.55)
        ramp.putpixel((x, 0), int(255 * min(1.0, a**1.4)))
    ramp = ramp.resize((S, S))
    blocks = Image.composite(blurred, blocks, ramp)

    blocks.putalpha(
        Image.composite(blocks.getchannel("A"), Image.new("L", (S, S), 0), inner_mask)
    )
    img.alpha_composite(blocks)

    # --- Grid hint: thin light lines over the study ---
    grid = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid)
    line_w = max(1, int(S * 0.006))
    for f in (1 / 3, 2 / 3):
        gx = box[0] + int(bw * f)
        gy = box[1] + int(bh * f)
        gd.line((gx, box[1], gx, box[3]), fill=(255, 255, 255, 150), width=line_w)
        gd.line((box[0], gy, box[2], gy), fill=(255, 255, 255, 150), width=line_w)
    grid.putalpha(
        Image.composite(grid.getchannel("A"), Image.new("L", (S, S), 0), inner_mask)
    )
    img.alpha_composite(grid)

    # --- Inner canvas border for a framed look ---
    frame = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(frame).rounded_rectangle(
        box,
        radius=inner_radius,
        outline=(255, 255, 255, 210),
        width=max(1, int(S * 0.012)),
    )
    img.alpha_composite(frame)

    # --- Apply the outer rounded-square mask ---
    outer = rounded_mask(S, radius)
    img.putalpha(outer)

    # Subtle top highlight for depth.
    hi = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hi)
    hd.rounded_rectangle(
        (0, 0, S - 1, int(S * 0.5)), radius=radius, fill=(255, 255, 255, 26)
    )
    hi.putalpha(Image.composite(hi.getchannel("A"), Image.new("L", (S, S), 0), outer))
    img.alpha_composite(hi)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    os.makedirs(HERE, exist_ok=True)

    master = build_base(SIZE)
    png_path = os.path.join(HERE, "icon.png")
    master.save(png_path)
    print("wrote", png_path)

    # --- Build iconset ---
    iconset = os.path.join(HERE, "icon.iconset")
    os.makedirs(iconset, exist_ok=True)
    specs = [
        (16, 1),
        (16, 2),
        (32, 1),
        (32, 2),
        (128, 1),
        (128, 2),
        (256, 1),
        (256, 2),
        (512, 1),
        (512, 2),
    ]
    for base, scale in specs:
        px = base * scale
        # Redraw small sizes from scratch for legibility rather than
        # downscaling the 1024 master (keeps edges crisp at 16px).
        im = build_base(px) if px <= 128 else master.resize((px, px), Image.LANCZOS)
        name = f"icon_{base}x{base}{'' if scale == 1 else '@2x'}.png"
        im.save(os.path.join(iconset, name))
    print("wrote iconset ->", iconset)

    icns = os.path.join(HERE, "icon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
    print("wrote", icns)

    # --- Preview strip of small sizes for inspection ---
    prev_sizes = [16, 32, 64, 128]
    pad = 24
    gap = 24
    bg = (250, 250, 250, 255)
    total_w = sum(prev_sizes) + gap * (len(prev_sizes) - 1) + pad * 2
    total_h = max(prev_sizes) + pad * 2
    preview = Image.new("RGBA", (total_w, total_h), bg)
    x = pad
    for s in prev_sizes:
        im = build_base(s)
        preview.alpha_composite(im, (x, pad + (max(prev_sizes) - s)))
        x += s + gap
    prev_path = os.path.join(HERE, "icon_preview.png")
    preview.save(prev_path)
    print("wrote", prev_path)

    # Clean up intermediate iconset.
    for f in os.listdir(iconset):
        os.remove(os.path.join(iconset, f))
    os.rmdir(iconset)
    print("removed", iconset)


if __name__ == "__main__":
    main()
