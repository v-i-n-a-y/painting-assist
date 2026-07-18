# Painting Assist

A coarse to fine painting reference tool. You load a reference photo and change
how it is shown on screen, never the original, so you can paint from a
deliberately processed view of it. The idea mirrors what painters already do at
the easel, squinting to collapse a scene into its big value masses before any
detail goes in.

Nothing is destructive. The original photo is held untouched in memory, the
controls apply as a live pipeline on top of it, Reset restores the clean view,
and Save writes out the processed image you are currently looking at.

## Controls

### Blur, block coarse to fine

The main control, in two modes.

- Continuous is a reversed Detail slider. Far left is a heavy blur, so you
  block in values and shapes first, and dragging back towards zero reveals fine
  detail as the painting progresses. More progress maps to moving right.
- Stepped is a fixed ladder of discrete blur set-points that you step through
  with Prev and Next (or the jump slider). Choose how many stages you want, then
  either let Even space them from the heaviest blur at Stage 1 down to sharp, or
  use Manual and type each stage's blur level yourself. Working one stage at a
  time keeps you honest, since you have to settle the big blocking at Stage 1
  before any detail is allowed in.

### Flip, check your drawing

Mirrors the reference horizontally, the oldest trick for catching drawing
errors. Toggle it from the Flip section in the dock or with the F key. A
vertical flip is available too. Because it sits in the pipeline everything
downstream, including the eyedropper and Save, sees the flipped image.

### Tone, judge temperature and contrast

Global contrast, saturation and temperature knobs, all neutral at zero. Tone
runs early in the pipeline, right after the crop, so the value and colour-group
tools further down see the adjusted image. Warm and cool shifts happen in Lab
space, which keeps them perceptually honest.

### Colour groups, flat colour masses

Reduces the reference to a handful of flat colour masses with Lab k-means, much
like a poster study. Set the number of colours and an optional smooth to
consolidate speckle before grouping. The colours it settles on show up as
clickable swatches in the Palette panel below the image. Click one to copy its
hex and read its value percentage, hue angle and chroma.

### Values, the value story

- Greyscale replaces colour with a neutral grey of the same perceptual
  lightness, giving a pure value study. Toggle it at any time with V.
- Value steps posterizes lightness into 2 to 8 flat bands, with 2 or 3 giving a
  notan. Keep colour posterizes the values but holds each pixel's hue, which is
  the value-grouped colour that helps in the mid layers.
- Isolate band dims everything outside one chosen value band, so you can study a
  single value mass on its own.

### Eyedropper, read any colour and match a mix

Press I (or use the toolbar button), then click the image to read the colour you
are actually looking at. The hex, RGB, value percentage, hue and chroma appear
in the status bar and the Palette panel. Set the sample size in the Palette
panel to average a small area rather than one noisy pixel, which matters when
you are matching a colour for the mid layers. Alongside the reading the panel
suggests an approximate mix from a limited palette, for example the Zorn earth
set, as a rough starting point rather than a precise recipe. Clicking any
colour-group swatch gives the same suggestion for that swatch, and Export
Palette under the File menu writes the swatches out as a PNG strip.

### Measure and proportion

Under View then Measure you get three drag-on-the-image tools. Angle reports the
angle of a line from horizontal for sight-size checks. Caliper compares one
length against another and shows the ratio. Guides drops a plumb and a horizon
line for checking verticals and horizontals. The readout follows in the status
bar as you drag.

### Grid, position by the grid method

An even columns by rows grid to transfer proportions and placement onto a
matching grid on your canvas. Pick the colour, opacity and line width, and add
corner-to-corner diagonals if you want to find the centre. The grid is a viewer
overlay, so it stays crisp at any zoom and is never baked into the processed
pixels. When you Save or export with it showing, the app asks whether to draw it
into the file, so you can still print a gridded reference.

### Canvas and Crop, match your surface

Enter your canvas width and height in any unit, since only the ratio matters.
With Lock to canvas ratio on, click Adjust crop region and an aspect-locked box
appears over the reference. Drag it to move, drag the corners to resize while it
holds your canvas proportions, then click Apply crop. Turn the lock off for a
freeform crop of any shape, and use Clear crop to restore the full frame. The
crop is stored as fractions of the image, so it is resolution-independent and
fully non-destructive.

## Install

Prebuilt installers for each release are on the
[Releases](https://github.com/v-i-n-a-y/painting-assist/releases) page.

| Platform      | Download                                                       |
| ------------- | ------------------------------------------------------------- |
| macOS (Apple) | `PaintingAssist-<version>-macos-arm64.dmg`                     |
| macOS (Intel) | `PaintingAssist-<version>-macos-x86_64.dmg`                    |
| Windows       | `PaintingAssist-<version>-windows-x64-setup.exe`              |
| Linux         | `PaintingAssist-<version>-linux-x86_64.AppImage` (or `.tar.gz`)|

- On macOS, open the DMG and drag Painting Assist to Applications. The build is
  unsigned, so on first launch you need to right-click the app and choose Open.
- On Windows, run the setup `.exe`. It installs to Program Files with Start Menu
  and optional desktop shortcuts.
- On Linux, mark the AppImage executable with `chmod +x` and run it, or extract
  the `.tar.gz` and run the `PaintingAssist` binary inside.

To run from source instead, see [Running](#running) below.

Building installers locally and the tagged-release flow are documented in
[`packaging/README.md`](packaging/README.md).

## Requirements

- [`uv`](https://docs.astral.sh/uv/) for environment management.
- Python pinned to 3.12 via `.python-version`. Dependencies install with
  `uv sync` and cover PySide6 (pinned to 6.8.1.1), numpy, opencv-python and
  Pillow.

Two macOS and Qt pins matter here.

- Python 3.12, set in `.python-version`. The uv-provided standalone 3.13 build
  fails to load Qt's `cocoa` platform plugin with this PySide6, so the GUI does
  not start. Leaving `.python-version` in place stops uv silently switching to
  3.13 and leaving you with a window that never opens.
- PySide6 6.8.1.1. Version 6.11.x fails to load its Qt platform plugins under the
  bundled interpreters.

If the app ever refuses to launch, with `cocoa` plugin errors or a
`ModuleNotFoundError: painting_assist`, the virtual environment has usually
drifted. Rebuild it with `rm -rf .venv && uv sync`.

## Running

```bash
uv sync                 # create the venv and install everything
uv run painting-assist  # launch the app
```

The toolbar carries Open (Ctrl+O), Save (Ctrl+S), Export Blur Steps (Ctrl+E),
Fit (Ctrl+0), 1:1 (Ctrl+1), Eyedropper (I), the measure tools, Undo (Ctrl+Z),
Redo (Ctrl+Shift+Z) and Reset. Undo and redo step through your control changes,
one edit at a time. The File menu adds Open Recent, Settings, and presets: Save
Preset stores your current control settings under a name, and Apply Preset
brings them back on any reference, which is handy for a repeatable underpainting
recipe. The Help menu lists every shortcut, including hold B for before and
after, V for the Values toggle and F for Flip. Your window layout, control
settings and last image are all restored on the next launch.

Settings lives under File and lets you pick the theme, either matching the
system or forcing light or dark, and choose how often the app checks for updates,
from every launch through to never. When a newer release is found the app offers
to download the right installer for your platform and open it, and Check for
Updates under Help runs the same check on demand.

Export Blur Steps writes the whole coarse to fine progression in one go. Set the
Blur control to Stepped mode, press the button, choose a folder, and you get one
PNG per stage (`blur_step_01_of_05_blur080.png` and so on), each with your crop
applied and the grid drawn in if you asked for it, from the heaviest blocking
through to sharp.

In the image view, scroll to zoom about the cursor and drag to pan. The view
survives slider re-renders, so it never jumps while you work. Panning pauses
while you are adjusting a crop region, so the crop box keeps the mouse.

## Responsiveness

Slider drags stay fluid because the render layer (`render_controller.py`)
debounces input, runs the pipeline on a worker thread, renders a downscaled
preview while you drag and a full-resolution pass on release, and drops stale
frames so only the newest result reaches the screen.

## Tests

```bash
uv run pytest                           # headless suite (pipeline, cache, controls)
uv run python tests/test_gui_smoke.py   # builds the window and drives the crop flow
```

## Adding a new control

A new control is normally one new file plus one import line, with no edits to the
pipeline, the panel, the renderer, the view or the main window. Each control
declares its knobs as a list of `Param` descriptors and the dock builds the right
widgets (slider, spin box, checkbox, combo box or text field) automatically.

1. Create `painting_assist/controls/<your_control>.py`.

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
       order = 20              # pipeline and panel order; lower runs first

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
   `painting_assist/controls/__init__.py`.

   ```python
   from . import posterize  # noqa: F401  -- registers PosterizeControl
   ```

### Custom editors

Some controls need richer UI than the auto-generated sliders, such as the Blur
stepper or the interactive crop tool. For those, override
`Control.create_editor(self, parent=None)` to return a widget that exposes a
`paramChanged(str, object)` signal, an `interaction(bool)` signal and a
`refresh()` method. Return `None`, the default, to fall back to the generic Param
UI. The Blur and Crop controls are worked examples.

The generic `ParamType` set is `INT`, `FLOAT`, `BOOL`, `CHOICE` and `TEXT`.

## Licence

Released under the MIT Licence. See [`LICENSE`](LICENSE) for the full text.
