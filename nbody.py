import rebound
import numpy as np
import h5py

# 시뮬레이션 초기화
sim = rebound.Simulation()
sim.add(m=1.0)  # 태양
sim.add(m=1e-3, a=1.0)  # 행성1
sim.add(m=1e-3, a=1.5)  # 행성2

sim.dt = 0.01
n_steps = 500

data = []

for _ in range(n_steps):
    sim.integrate(sim.t + sim.dt)
    positions = []
    for p in sim.particles:
        positions.extend([p.x, p.y, p.vx, p.vy])
    data.append(positions)

data = np.array(data)

# HDF5 저장
with h5py.File("nbody_data.h5", "w") as f:
    f.create_dataset("positions", data=data)