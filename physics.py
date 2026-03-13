import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)


# --- COLOR PALETTE (hex) ---
COLOR_WATER        = 0x068FFF   # Blue  – freefall / no magnet
COLOR_CORE         = 0xFFD700   # Gold  – inside bubble radius
COLOR_MIDDLE_RING  = 0xFF8C00   # Orange– just outside bubble radius
COLOR_OUTER_RING   = 0x8B0000   # Dark red – far from center


@ti.data_oriented
class FluidSimulation:
    def __init__(self, num_particles: int = 10_000):
        self.n_particles = num_particles

        # Per-particle GPU fields
        self.pos   = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.vel   = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.color = ti.field(dtype=int,   shape=self.n_particles)

        # Shared control state (written from CPU, read on GPU)
        self.target           = ti.Vector.field(2, dtype=float, shape=())
        self.is_magnet_active = ti.field(dtype=int, shape=())

        self.target[None]           = [0.5, 0.5]
        self.is_magnet_active[None] = 0

        # Exponential smoothing state (CPU-side, applied before GPU write)
        self._smooth_cx = 0.5
        self._smooth_cy = 0.5
        self._alpha      = 0.2   # 0 = no movement, 1 = no smoothing

        self._init_particles()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @ti.kernel
    def _init_particles(self):
        for i in self.pos:
            self.pos[i]  = [ti.random() * 0.5 + 0.25, ti.random() * 0.5 + 0.25]
            self.vel[i]  = [0.0, 0.0]
            self.color[i] = COLOR_WATER

    # ------------------------------------------------------------------
    # CPU-side control update  (called once per frame from main.py)
    # ------------------------------------------------------------------

    def update_control(self, cx: float, cy: float, is_open: bool):
        """
        Smooth the incoming hand position with an exponential moving average
        before writing it to the GPU, so jittery tracking doesn't feed
        directly into the physics.
        """
        a = self._alpha
        self._smooth_cx = a * cx        + (1 - a) * self._smooth_cx
        self._smooth_cy = a * (1 - cy)  + (1 - a) * self._smooth_cy  # flip Y once here

        self.target[None]           = [self._smooth_cx, self._smooth_cy]
        self.is_magnet_active[None] = 1 if is_open else 0

    # ------------------------------------------------------------------
    # GPU kernel  (called once per frame from main.py)
    # ------------------------------------------------------------------

    @ti.kernel
    def step(self, dt: float):
        """
        Advance the simulation by `dt` seconds.
        dt is passed in from main.py so physics scale correctly
        with actual frame time instead of an assumed 60 fps.
        """

        # --- PHYSICS TUNING PANEL ---
        spring_stiffness = 150.0  # Restoring force toward bubble surface
        bubble_radius    = 0.05   # Radius of the vortex ball
        jitter_strength  = 15.0   # Brownian noise — stops particles from freezing
        air_resistance   = 0.85   # Velocity damping per frame
        gravity          = -15.0  # Downward acceleration when magnet is off
        # ----------------------------

        for i in self.pos:
            if self.is_magnet_active[None] == 1:
                direction = self.target[None] - self.pos[i]
                distance  = direction.norm()

                if distance > 0.0001:
                    radial_dir = direction / distance

                    # 1. HOOKE'S LAW  – spring force toward/away from bubble surface
                    #    displacement > 0  →  particle is outside → pulled in
                    #    displacement < 0  →  particle is inside  → pushed out
                    displacement  = distance - bubble_radius
                    spring_force  = radial_dir * (displacement * spring_stiffness)

                    # 2. BROWNIAN MOTION  – random kick so particles never stack perfectly
                    jitter = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * jitter_strength

                    self.vel[i] += (spring_force + jitter) * dt

                else:
                    # 3. ESCAPE HATCH  – if a particle lands exactly on the center,
                    #    blast it outward randomly to avoid a degenerate zero-vector
                    explosion    = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 100.0
                    self.vel[i] += explosion * dt

            else:
                # No magnet → simple gravity drop
                self.vel[i] += ti.Vector([0.0, gravity]) * dt

            # Damping + integration
            self.vel[i]  *= air_resistance
            self.pos[i]  += self.vel[i] * dt

            # --- COLOR BY DISTANCE FROM VORTEX CENTER ---
            if self.is_magnet_active[None] == 1:
                dist = (self.target[None] - self.pos[i]).norm()
                if dist < bubble_radius + 0.01:
                    self.color[i] = COLOR_CORE
                elif dist < bubble_radius + 0.04:
                    self.color[i] = COLOR_MIDDLE_RING
                else:
                    self.color[i] = COLOR_OUTER_RING
            else:
                self.color[i] = COLOR_WATER

            # --- WALL COLLISIONS ---
            # ti.static(range(2)) unrolls the loop at compile time (Taichi idiom),
            # which lets us treat both X and Y axes identically without branching overhead.
            for axis in ti.static(range(2)):
                if self.pos[i][axis] < 0.02:
                    self.pos[i][axis]  = 0.02
                    self.vel[i][axis] *= -0.1
                elif self.pos[i][axis] > 0.98:
                    self.pos[i][axis]  = 0.98
                    self.vel[i][axis] *= -0.1