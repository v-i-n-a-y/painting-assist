# Packaging

Native installers for Painting Assist are built with
[PyInstaller](https://pyinstaller.org) (onedir, windowed) and platform-specific
packaging tools. The shared build spec is `packaging/painting_assist.spec` and
the frozen entry point is `packaging/launch.py`.

The app version is currently `0.10.0`, kept in three places that must be in sync:

- `pyproject.toml` (`version`)
- `painting_assist/__init__.py` (`__version__`)
- `packaging/painting_assist.spec` (`APP_VERSION`) and
  `packaging/windows_installer.iss` (the `AppVersion` fallback).

CI installer filenames are no longer hardcoded: `build-installers.yml` derives
the version from the release tag (`v0.10.0` -> `0.10.0`), falling back to
`__init__.py` on manual dispatch, and passes it to Inno Setup via
`iscc /DAppVersion=...`. So a release only needs a matching `v*` tag; the three
in-repo spots above just keep the app's own self-reported version correct.

## Local builds

All commands run from the repository root with dev dependencies synced:

```sh
uv sync --dev
```

### macOS (.app + .dmg)

```sh
uv run pyinstaller packaging/painting_assist.spec --noconfirm
# -> dist/Painting Assist.app

# Strip xattrs, ad-hoc sign, then build a compressed DMG with an
# /Applications drop target:
xattr -cr "dist/Painting Assist.app"
codesign -s - --force --deep "dist/Painting Assist.app"
STAGE="$(mktemp -d)"
cp -R "dist/Painting Assist.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Painting Assist" -srcfolder "$STAGE" \
  -ov -format UDZO dist/PaintingAssist-0.10.0-macos-arm64.dmg
rm -rf "$STAGE"
```

The DMG is unsigned/unnotarised, so first launch needs right-click -> Open (or
`xattr -dr com.apple.quarantine` on the installed app).

### Windows (.exe installer)

```sh
uv run pyinstaller packaging/painting_assist.spec --noconfirm
# -> dist/Painting Assist/PaintingAssist.exe

iscc packaging\windows_installer.iss
# -> dist/PaintingAssist-0.10.0-windows-x64-setup.exe
```

Requires [Inno Setup](https://jrsoftware.org/isinfo.php) (`iscc` on PATH). It is
preinstalled on GitHub `windows-latest` runners.

### Linux (AppImage + tar.gz)

```sh
uv run pyinstaller packaging/painting_assist.spec --noconfirm
# -> dist/Painting Assist/PaintingAssist
```

The AppImage is assembled by the CI workflow (AppDir + `.desktop` + icon, then
`appimagetool`). See the `linux` job in
`.github/workflows/build-installers.yml` for the exact steps, including the Qt
xcb runtime libraries that must be installed.

## CI / release flow

`.github/workflows/build-installers.yml` builds every platform on:

- **`workflow_dispatch`** — manual runs that only upload build artifacts, for
  testing the build legs.
- **push of a `v*` tag** — additionally attaches the artifacts to the matching
  GitHub release via `softprops/action-gh-release`.

To cut a release:

```sh
# bump the version in the three places listed above, commit, then:
git tag v0.10.0
git push origin v0.10.0
```

The matrix produces:

| Platform        | Artifact                                        |
| --------------- | ----------------------------------------------- |
| macOS arm64     | `PaintingAssist-0.10.0-macos-arm64.dmg`          |
| macOS x86_64    | `PaintingAssist-0.10.0-macos-x86_64.dmg`         |
| Windows x64     | `PaintingAssist-0.10.0-windows-x64-setup.exe`    |
| Linux x86_64    | `PaintingAssist-0.10.0-linux-x86_64.AppImage`    |
| Linux x86_64    | `PaintingAssist-0.10.0-linux-x86_64.tar.gz`      |
