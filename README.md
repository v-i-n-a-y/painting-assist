# Painting Assist

A coarse-to-fine painting **reference** tool. You load a reference photo and
manipulate how it is *displayed* — never the original — so you can paint from a
deliberately processed view of it. The name is what painters do: squint to
collapse a scene into its big value masses before refining detail.

Everything is non-destructive: the original is held untouched in memory,
controls are applied as a live pipeline on top of it, **Reset** restores the
clean view, and **Save** writes out the *processed* image (what you currently
see).

## Controls

### Blur — block coarse → fine
The flagship control, in two modes:

- **Continuous** — a reversed **Detail** slider. Left = heavy blur (block in
  values and shapes first); drag right toward 0 to reveal fine detail as your
  painting progresses. "More progress" maps to "move right".
- **Stepped** — a fixed ladder of discrete blur **set-points** you step through
  with **◀ Prev / Stage k of N / Next ▶** (or the jump slider). Choose the
  number of stages, then either:
  - **Even** — set the heaviest blur (Stage 1) and the rest are spaced evenly
    down to sharp; or
  - **Manual** — type each stage's blur level into its own box.

  Working stage by stage keeps you honest: nail the big blocking at Stage 1
  before letting any detail in.

### Tone — judge temperature and contrast
Global **contrast**, **saturation** and **temperature** knobs, all neutral at
zero. Runs early in the pipeline (right after the crop), so the value and
colour-group tools below see the adjusted image. Warm/cool shifts are done in
Lab space, which keeps them perceptually honest.

### Colour groups — flat colour masses
Reduces the reference to a handful of flat colour masses (Lab k-means), like a
poster study — set the number of **colours** and an optional **smooth** to
consolidate speckle before grouping. The colours it finds appear as clickable
swatches in the **Palette** panel below the image: click one to copy its hex
and read its value %, hue angle and chroma.

### Values — the value story
- **Greyscale** — replace colour with neutral grey of the same perceptual
  lightness, a pure value study. Toggle it any time with **V**.
- **Value steps** — posterize lightness into 2–8 flat bands (2–3 = notan);
  **Keep colour** posterizes the values but keeps each pixel's hue
  (value-grouped colour, useful in the mid layers).
- **Isolate band** — dim everything outside one chosen value band so you can
  study a single value mass at a time.

### Eyedropper — read any colour
Press **I** (or the toolbar button), then click the image to read the colour
you are actually looking at — hex, RGB, value %, hue and chroma appear in the
status bar and the Palette panel. Handy for checking a mix against the
processed reference.

### Grid — position by the grid method
An even **columns × rows** grid to transfer proportions and placement onto a
matching grid on your canvas. Pick the colour, opacity and line width, and
optionally add corner-to-corner **diagonals** to find the centre. The grid is a
**viewer overlay** — crisp at any zoom, never baked into the processed pixels —
and when you Save or export with it showing, you're asked whether to draw it
into the file (so you can still print a gridded reference).

### Canvas & Crop — match your surface
Enter your canvas' **width × height** (any unit — only the ratio matters). With
**Lock to canvas ratio** on, click **Adjust crop region…** and an aspect-locked
box appears over the reference: drag to move, drag the corners to resize (it
keeps your canvas proportions), then **Apply crop**. Turn the lock **off** for a
**freeform** crop of any shape. **Clear crop** restores the full frame. The crop
is stored as fractions of the image, so it is resolution-independent and fully
non-destructive.

## Install

Prebuilt installers for each release are on the
[**Releases**](https://github.com/v-i-n-a-y/painting-assist/releases) page:

| Platform      | Download                                                       |
| ------------- | ------------------------------------------------------------- |
| macOS (Apple) | `PaintingAssist-<version>-macos-arm64.dmg`                     |
| macOS (Intel) | `PaintingAssist-<version>-macos-x86_64.dmg`                    |
| Windows       | `PaintingAssist-<version>-windows-x64-setup.exe`              |
| Linux         | `PaintingAssist-<version>-linux-x86_64.AppImage` (or `.tar.gz`)|

- **macOS** — open the DMG and drag *Painting Assist* to Applications. The build
  is unsigned, so on first launch right-click the app and choose **Open**.
- **Windows** — run the setup `.exe` (installs to Program Files, with Start Menu
  and optional desktop shortcuts).
- **Linux** — `chmod +x` the AppImage and run it, or extract the `.tar.gz` and
  run the `PaintingAssist` binary inside.

To run from source instead, see [Running](#running) below (`uv run painting-assist`).

Building installers locally and the tagged-release flow are documented in
[`packaging/README.md`](packaging/README.md).

## Requirements

- [`uv`](https://docs.astral.sh/uv/) for environment management
- Python is pinned to **3.12** via `.python-version`. Dependencies install via
  `uv sync`: PySide6 (pinned to **6.8.1.1**), numpy, opencv-python, Pillow

> **macOS / Qt note — two pins that matter:**
> - **Python 3.12** (`.python-version`). The uv-provided standalone **3.13**
>   build fails to load Qt's `cocoa` platform plugin with this PySide6, so the
>   GUI won't start. Don't delete `.python-version`, or uv will silently switch
>   to 3.13 and the window won't open.
> - **PySide6 6.8.1.1**. 6.11.x fails to load its Qt platform plugins under the
>   bundled interpreters.
>
> If the app ever won't launch (plugin/`cocoa` errors, or
> `ModuleNotFoundError: painting_assist`), the venv has usually drifted — rebuild
> it: `rm -rf .venv && uv sync`.

## Running

```bash
uv sync                 # create the venv and install everything
uv run painting-assist  # launch the app
```

Toolbar: **Open** (Ctrl+O), **Save** (Ctrl+S), **Export Blur Steps** (Ctrl+E) ·
**Fit** (Ctrl+0), **1:1** (Ctrl+1) · **Eyedropper** (I) · **Reset**. The File
menu adds **Open Recent** and the Help menu lists every shortcut (hold **B**
for before/after, **V** for the Values toggle). Your window layout, control
settings and last image are restored on the next launch.

**Export Blur Steps** writes the whole coarse→fine progression in one click: set
the Blur control to **Stepped** mode, press the button, choose a folder, and you
get one PNG per stage (`blur_step_01_of_05_blur080.png` …), each with your crop
applied (and the grid drawn in, if you say yes when asked) — heaviest blocking
through to sharp.

In the image view, scroll to zoom about the cursor and
drag to pan; the view survives slider re-renders so it never jumps while you
work. (Panning is paused while you are adjusting a crop region so the crop box
gets the mouse.)

## Responsiveness

Slider drags stay fluid because the render layer (`render_controller.py`)
debounces input, runs the pipeline on a worker thread, renders a downscaled
preview *while dragging* and a full-resolution pass on release, and drops stale
frames so only the newest result is shown.

## Tests

```bash
uv run pytest                           # headless suite (pipeline, cache, controls)
uv run python tests/test_gui_smoke.py   # builds the window, drives the crop flow — PASS/FAIL
```

## Adding a new control

A new control is normally **one new file plus one import line** — no edits to
the pipeline, the panel, the renderer, the view, or the main window. Each
control declares its knobs as a list of `Param` descriptors and the dock builds
the right widgets (slider / spin box / checkbox / combo box / text field)
automatically.

1. Create `painting_assist/controls/<your_control>.py`:

   ```python
   from __future__ import annotations

   from typing import List

   import numpy as np

   from painting_assist.controls.base import Control, Param, ParamType
   from painting_assist.controls.registry import register


   @register
   class PosterizeControl(Control):
       id = "posterize"        # stable unique key (used in saved state)
       name = "Posterize"      # shown as the dock section title
       order = 20              # pipeline + panel order; lower runs first

       @classmethod
       def params(cls) -> List[Param]:
           return [Param(name="levels", label="Value levels", ptype=ParamType.INT,
                         default=8, minimum=2, maximum=32, step=1)]

       def is_active(self) -> bool:
           return self.enabled and int(self.get("levels")) < 256

       def process(self, img: np.ndarray) -> np.ndarray:
           n = int(self.get("levels"))
           s = 255.0 / (n - 1)
           return (np.round(img / s) * s).astype(np.uint8)
   ```

2. Register it for discovery by adding one line to
   `painting_assist/controls/__init__.py`:

   ```python
   from . import posterize  # noqa: F401  -- registers PosterizeControl
   ```

### Custom editors

For controls that need richer UI than auto-generated sliders (like Blur's
stepper or the interactive crop tool), override
`Control.create_editor(self, parent=None)` to return a widget that exposes a
`paramChanged(str, object)` signal, an `interaction(bool)` signal, and a
`refresh()` method. Return `None` (the default) to use the generic Param UI.
The Blur and Crop controls are worked examples.

The generic `ParamType` set is `INT`, `FLOAT`, `BOOL`, `CHOICE`, `TEXT`.
