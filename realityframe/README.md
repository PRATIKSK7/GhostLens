# 👻 PhantomFrame - AI Vision System

## Setup
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Controls
| Key | Action |
|-----|--------|
| 1 | Normal mode |
| 2 | Partial invisibility (50%) |
| 3 | Full invisibility |
| F | Toggle Focus Box mode |
| R | Reset region selection |
| A | Toggle Face Anchor (keep face visible) |
| V | Start/Stop video recording |
| D | Toggle MiDaS AI depth mode |
| Q | Quit |

## Gestures
| Gesture | Effect |
|---------|--------|
| Pinch (thumb+index) | Partial invisibility |
| Open left palm | Full invisibility |

## Focus Box Mode
1. Press F to enter focus mode
2. Click 4 points on the screen to define visible region
3. Everything outside the box is replaced with background
4. Press R to reset and reselect region

## Recording
- Press V to start recording
- Red REC indicator shows in top-left
- Press V again to stop and save
- Files saved to recordings/ folder as MP4
