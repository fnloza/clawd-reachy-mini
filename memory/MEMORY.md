# clawd-reachy-mini Memory

## Running Instructions

```bash
uv run clawd-reachy \
  --gateway-host 127.0.0.1 \
  --gateway-port 18789 \
  --gateway-token "<token>"
```

## Environment Setup (macOS / Apple Silicon)

### Python Version
- Must use **Python 3.13** (not 3.12 or 3.14)
- `libusb-package` (reachy_mini dependency) only supports cp310–cp313
- uv's standalone Python won't work — must use **Homebrew's Python 3.13**

```bash
brew install python@3.13
rm -rf .venv
uv venv --python /opt/homebrew/bin/python3.13
uv sync
```

### GStreamer / PyGObject (`gi` module)
- `gi` cannot be pip-installed on macOS — comes from Homebrew
- `pygobject3` must be installed via brew AND compiled for the same Python version as the venv

```bash
brew install pygobject3 gstreamer gst-plugins-base gst-plugins-good \
     gst-plugins-bad gst-plugins-ugly gst-libav

# Link gi into the venv (brew compiles for 3.13 and 3.14)
ln -s /opt/homebrew/lib/python3.13/site-packages/gi \
  .venv/lib/python3.13/site-packages/gi

# Verify
uv run python -c "from gi.repository import GLib; print('GLib OK')"
```

- If the symlink already exists but is dangling, check which Python version brew compiled gi for:
  `find $(brew --prefix)/lib -name "gi" -type d`

## Known Issues / Debugging

### Simulation mode despite daemon running
- Root cause: broad `except ImportError` in `_connect_reachy` was catching ImportErrors from inside `ReachyMini()` init (e.g. missing `gi`), not just from the `reachy_mini` import itself
- Fix: split import guard from instantiation (already applied in [interface.py](../src/clawd_reachy_mini/interface.py))

### GLib dylib not found (`libgobject-2.0.0.dylib`)
- Happens when using uv's standalone Python — it doesn't know about brew's dylib paths
- Fix: use brew's Python 3.13 for the venv (see above)

### Gateway warning: "binding to a non-loopback address"
- This is from the OpenClaw gateway daemon, not our code
- Safe to ignore when `--gateway-token` is set (auth is configured)
- To suppress: set gateway bind address to `127.0.0.1` in gateway config

## Architecture

- `interface.py` — main loop: connect robot → STT → gateway → TTS → animate
- `audio.py` — mic capture with VAD (energy threshold), supports sounddevice or Reachy WebRTC mic
- `stt.py` — Whisper / faster-whisper / OpenAI cloud backends
- `gateway.py` — WebSocket client for OpenClaw protocol (connect challenge → chat.send → agent events)
- `elevenlabs.py` — TTS via ElevenLabs API
- `config.py` — all settings, CLI args override env vars
