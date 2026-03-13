import time

import cv2
import taichi as ti

from vision  import HandTracker
from physics import FluidSimulation

# --- CONSTANTS ---
WINDOW_TITLE   = "Aero-Dynamic Fluid Controller"
WINDOW_RES     = (800, 800)
NUM_PARTICLES  = 10_000
PARTICLE_RADIUS = 2.0
MAX_DT         = 1 / 30   # Cap dt at ~30 fps equivalent to avoid physics explosion on lag spikes
QUIT_KEY       = ord('q')


def main():
    # ------------------------------------------------------------------
    # 1. Initialise webcam  – fail loudly if not available
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam on device index 0. "
            "Check that a camera is connected and not in use by another app."
        )

    # ------------------------------------------------------------------
    # 2. Initialise vision and physics
    # ------------------------------------------------------------------
    tracker = HandTracker()
    fluid   = FluidSimulation(num_particles=NUM_PARTICLES)

    # ------------------------------------------------------------------
    # 3. Create the Taichi render window
    # ------------------------------------------------------------------
    gui = ti.GUI(WINDOW_TITLE, res=WINDOW_RES)

    print("System Online!")
    print("  Open hand  → Doctor Strange vortex")
    print("  Closed fist → Gravity drop")
    print("  Press 'q' in the webcam window to quit.")

    last_time = time.perf_counter()

    # ------------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------------
    while gui.running:

        # --- Frame timing (real dt, capped to avoid instability on lag) ---
        now     = time.perf_counter()
        dt      = min(now - last_time, MAX_DT)
        last_time = now

        # --- Vision ---
        success, frame = cap.read()
        if not success:
            print("Warning: failed to grab webcam frame — skipping.")
            continue

        frame = cv2.flip(frame, 1)                   # Mirror for natural interaction
        state = tracker.get_hand_state(frame)         # Returns a HandState dataclass

        # --- Physics ---
        fluid.update_control(state.cx, state.cy, state.is_open)
        fluid.step(dt)

        # --- Render particles ---
        gui.circles(
            fluid.pos.to_numpy(),
            radius=PARTICLE_RADIUS,
            color=fluid.color.to_numpy(),
        )
        gui.show()

        # --- Debug webcam feed ---
        cv2.imshow("Hand Tracking Debug", state.frame)
        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            break

    # ------------------------------------------------------------------
    # 5. Cleanup
    # ------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()