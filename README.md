# ⚡ MotionForge — Universal AI Motion Controls

Play virtually any PC game with your body. MotionForge watches you through
your webcam, understands your movements with real-time pose estimation,
figures out what game you're playing and what its controls mean, designs
intuitive full-body gestures for those controls with AI, and injects
ordinary keyboard/mouse input — no game mods, no plugins, no VR headset.

```
camera → pose estimation → gesture recognition → semantic actions → SendInput
   ↑                                                    ↑
 multi-cam                    game detection → control discovery → AI mapping
```

## Quick start

```bat
pip install -r requirements.txt
run.bat
```

1. **Calibrate** — Settings ▸ *Run calibration*, stand still 3 seconds.
2. **Launch a game** (Minecraft, Roblox, Krunker in a browser tab, …) —
   MotionForge detects it and loads/creates the right profile automatically.
3. **Arm** — press **F9** or hold a **T-pose** for 1 second. The dot in the
   camera view turns green.
4. Move. March in place to walk, jump to jump, swing your arm to attack.

**Safety:** F9 (configurable) and the T-pose always toggle motion control.
Input is only injected while the game window has focus, and every held key is
released the instant you disarm, switch windows, or leave the frame.

Run the end-to-end diagnostic at any time:

```bat
run.bat --selftest
```

## What's inside

| Layer | What it does |
|---|---|
| **Game detection** | Foreground-window polling, process knowledge base (Steam/Epic/Xbox/Riot/Battle.net/Ubisoft/Roblox/Minecraft/emulators), Steam appmanifest lookup, browser-tab title matching for web games, fullscreen heuristic, and Gemini screenshot identification for unknown titles. |
| **Control discovery** | Parses Minecraft `options.txt` exactly; scans install/AppData/Documents for keybind-shaped JSON/INI/XML/YAML/cfg files; manual capture wizard ("press the key for JUMP") as the universal fallback. Discovered binds are stored on the profile forever. |
| **Semantic understanding** | Raw binds become semantic actions (`jump`, `reload`, `heal`, …) via a pattern engine, with Gemini classifying anything cryptic (`ThrowFrag` → `grenade`). |
| **AI gesture mapping** | Gemini (with your key in `.env`) designs mappings per game with a one-line rationale for each ("Bringing a hand to the mouth is a natural gesture for consuming something to heal"). A built-in heuristic engine with per-genre templates (FPS, melee, sandbox, platformer, racing, pointer) covers offline use. Suggestions are reviewed in the UI before applying. |
| **Preference learning** | Remap a suggestion twice and MotionForge remembers — future suggestions (offline *and* AI) use your preferred gesture for that action. |
| **Pose estimation** | MediaPipe PoseLandmarker (Tasks API), 33 landmarks, lite/full/heavy models with automatic performance tuning to hold frame rate. Multi-camera failover when you're occluded. |
| **Gesture engine** | 25+ temporal detectors in body-relative units: punches, swings, chops, throws, push, clap, wave, hand-to-mouth/chest, block, bow draw, kicks, jump, crouch, leans (4 directions), walk/sprint-in-place, arm raises, arms-up, T-pose. Priority suppression ensures one motion = one action; dwell/hysteresis kill false positives. |
| **Continuous aim** | Head-look, lean-look, or hand-as-joystick for FPS camera control. Pointer games (Chess.com, GeoGuessr) use hand-cursor mode: reach your right hand toward the screen to steer the cursor, **hold it still ~1s to dwell-click**, jab your left fist to click instantly, or hold your left arm up to drag pieces. Click gestures never use the cursor hand, so clicking can't move your aim. |
| **Input injection** | Raw Win32 `SendInput` with hardware scancodes (DirectInput-safe), relative mouse movement at 120 Hz, tap/double/toggle/hold/hold-pulse semantics (repeated chops keep the mine button held in Minecraft). Games can't tell it from real hardware. |
| **Profiles** | Per-game JSON in `~/.motionforge/profiles`, auto-loaded on detection, unlimited variants (competitive / seated / fitness), editable in the Mappings tab. |
| **Accessibility** | Seated, one-handed (left/right), adjustable sensitivities; detector sets and suggestion templates adapt (seated: raise both arms = jump). |
| **Latency** | Capture→injection tracked live (EMA + p95 in the dashboard). Latest-frame-only capture, ~20–30 ms pose inference, sub-millisecond injection. |

## Configuration

- `.env` — `GEMINI_API_KEY=...` enables the AI features (screenshot game ID,
  semantic classification, mapping generation). Without it, everything still
  works using the offline heuristics. **Never commit this file.**
- `~/.motionforge/settings.json` — everything in the Settings tab, plus
  calibration. `~/.motionforge/profiles/*.json` — game profiles.
- `run.bat --dry-run` — full recognition with injection disabled (great for
  testing gestures without a game).

## Tests

```bat
python -m pytest tests -q        # 36 unit tests (synthetic pose streams)
run.bat --selftest               # 10 end-to-end checks incl. live camera + Gemini
```

## Controller emulation (future-ready)

`motionforge/inputs/gamepad.py` wraps a virtual Xbox 360 pad behind the same
binding interface (`pad:a`, `pad:rt`, …). To enable later: install the
[ViGEmBus driver](https://github.com/nefarius/ViGEmBus/releases) and
`pip install vgamepad`.

## Architecture notes for future expansion

The spec's long-term items map onto existing seams: finger tracking (swap the
pose model for MediaPipe Holistic in `vision/pose.py`), voice commands (a new
event source feeding `ActionExecutor`), smartwatch/haptics (new device layer
beside `inputs/`), cloud profile sync + community gesture libraries (the
profile JSON format is the interchange format), multiplayer synchronized
gestures (profiles + events are already serializable).
