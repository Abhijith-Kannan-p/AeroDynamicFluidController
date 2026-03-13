# 🌀 AeroDynamic Fluid Controller

A real-time GPU-accelerated fluid particle simulation controlled by your hand gestures via webcam. Open your hand to summon a Doctor Strange–style vortex that pulls 10,000 particles into a swirling orb. Close your fist and watch them rain down under gravity.


---

## ✨ Demo

| Open Hand — Vortex Mode | Closed Fist — Gravity Drop |
|:-:|:-:|
| Particles orbit your fingertip in a glowing orb | Particles fall freely under gravity |

> Particles are color-coded by distance from the vortex center:
> 🟡 **Gold** → core · 🟠 **Orange** → mid-ring · 🔴 **Dark Red** → outer field · 🔵 **Blue** → freefall

---

## 🏗️ Architecture

```
AeroDynamicFluidController/
├── main.py        # Entry point — event loop, frame timing, rendering
├── physics.py     # GPU fluid simulation via Taichi kernels
├── vision.py      # Hand tracking & gesture detection via MediaPipe
└── README.md
```

### How the three modules talk to each other

```
Webcam Frame
     │
     ▼
vision.py  ──► HandState(cx, cy, is_open, frame)
                         │
                         ▼
              physics.py ──► update_control()  ──► GPU fields
                         └──► step(dt)         ──► particle positions & colors
                                                         │
                                                         ▼
                                               main.py  gui.circles()
```

---

## ⚙️ How It Works

### Vision (`vision.py`)
- Uses **MediaPipe Hands** to detect a single hand in each webcam frame.
- Hand center is tracked via landmark `#9` (base of the middle finger).
- **Gesture detection** counts how many fingers are extended by comparing each fingertip Y-coordinate to its PIP (proximal interphalangeal) joint — robust to hand rotation and tilt.
- Raw landmark positions are returned in a `HandState` dataclass, preventing tuple-unpacking bugs.

### Physics (`physics.py`)
- Runs entirely on the **GPU** via **Taichi** (`@ti.data_oriented`, `@ti.kernel`).
- Each particle obeys **Hooke's Law** relative to a `bubble_radius` around the hand target:
  - Outside the radius → spring pulls it inward.
  - Inside the radius → spring pushes it outward.
- **Brownian jitter** keeps particles in constant motion so they never freeze into a static shell.
- An **escape hatch** handles the degenerate case where a particle lands at exact center (zero-length direction vector).
- When the magnet is off, particles experience simple **gravity**.
- Incoming hand coordinates are **exponentially smoothed** (EMA, α = 0.2) before being written to the GPU, so tracking jitter doesn't feed directly into the physics.

### Main Loop (`main.py`)
- Measures **real elapsed time** with `time.perf_counter()` and passes `dt` to `step()` so physics are frame-rate independent.
- `dt` is **capped at 1/30 s** to prevent physics explosion during lag spikes.
- Fails loudly at startup if no webcam is detected, rather than silently failing each frame.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10 or 3.11** (Taichi 1.7.x does not yet support 3.12)
- A CUDA-capable NVIDIA GPU (recommended) — falls back to CPU with `ti.init(arch=ti.cpu)`
- A webcam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AeroDynamicFluidController.git
cd AeroDynamicFluidController

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install taichi mediapipe opencv-python numpy
```

### Running

```bash
python main.py
```

Two windows will open:
- **Aero-Dynamic Fluid Controller** — the particle simulation (Taichi GUI)
- **Hand Tracking Debug** — your webcam feed with landmark overlay

Press **`q`** in the webcam window to quit.

---

## 🎮 Controls

| Gesture | Effect |
|---|---|
| ✋ Open hand (3+ fingers extended) | Activates vortex — particles spiral toward your fingertip |
| ✊ Closed fist | Deactivates vortex — particles fall under gravity |
| Move hand | Moves the vortex center in real time |

---

## 🔧 Tuning the Simulation

All physics parameters are grouped in a clearly labelled panel at the top of `physics.py`'s `step()` kernel:

```python
# --- PHYSICS TUNING PANEL ---
spring_stiffness = 150.0  # How aggressively particles orbit the bubble surface
bubble_radius    = 0.05   # Size of the vortex orb
jitter_strength  = 15.0   # Brownian noise — stops particles from freezing
air_resistance   = 0.85   # Velocity damping (0 = instant stop, 1 = no damping)
gravity          = -15.0  # Downward pull when magnet is off
```

Adjust the smoothing responsiveness in `FluidSimulation.__init__`:

```python
self._alpha = 0.2   # 0.0 = frozen, 1.0 = raw / no smoothing
```

Change the gesture sensitivity threshold in `vision.py`:

```python
OPEN_FINGER_THRESHOLD = 3   # Fingers that must be extended to trigger vortex
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `taichi` | 1.7.4 | GPU-accelerated particle physics |
| `mediapipe` | 0.10.x | Real-time hand landmark detection |
| `opencv-python` | 4.x | Webcam capture and frame display |
| `numpy` | 1.x | Array bridge between Taichi and OpenCV |

---

## 🐛 Troubleshooting

**`TaichiSyntaxError: Please decorate class FluidSimulation with @ti.data_oriented`**
The `@ti.data_oriented` decorator is required on any class containing `@ti.kernel` methods. Make sure it is present on the `FluidSimulation` class in `physics.py`.

**Webcam not detected**
Change the device index in `main.py` from `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` (or higher) if you have multiple cameras.

**Running without a CUDA GPU**
Change the init line in `physics.py`:
```python
ti.init(arch=ti.cpu)   # or ti.vulkan, ti.metal on macOS
```

**Gesture not triggering / too sensitive**
Adjust `OPEN_FINGER_THRESHOLD` in `vision.py`. Lower it (e.g. `2`) to trigger more easily; raise it (e.g. `4`) to require a fuller open hand.

---

## 🗺️ Roadmap

- [ ] Two-hand support — one hand per vortex
- [ ] Particle collision / pressure simulation
- [ ] Record and export simulation as video
- [ ] Configurable particle count and colors via CLI args
- [ ] Web demo via Taichi's GGUI or WebAssembly export

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
[README.md](https://github.com/user-attachments/files/25976873/README.md)
