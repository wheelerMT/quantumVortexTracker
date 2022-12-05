import pygpe.scalar as gpe
import vortices as vortex
import cupy as cp
import h5py
import numpy as np


def generate_data(iteration: int):
    if iteration % 100 == 0:
        print(f'On iteration {iteration}')
    vortex_pos = vortex.generate_positions(grid, 2, 2)
    x_pos, y_pos = zip(*vortex_pos)
    positions[iteration, :] = x_pos + y_pos
    initial_phase = vortex.vortex_phase_profile(grid, vortex_pos)
    psi = gpe.Wavefunction(grid)
    psi.set_wavefunction(cp.ones(grid_points, dtype='complex128'))
    psi.apply_phase(cp.asarray(initial_phase))
    psi.fft()
    for _ in range(params["nt"]):
        gpe.step_wavefunction(psi, params)
    psi.ifft()

    # Reshape before saving
    phase = cp.asnumpy(cp.angle(psi.wavefunction))
    phase = phase[..., np.newaxis]
    phases[iteration, ...] = phase


grid_points = (256, 256)
grid_spacing = (0.5, 0.5)
grid = gpe.Grid(grid_points, grid_spacing)
params = {
    "g": 1,
    "trap": 0,
    "nt": 100,
    "dt": -1j * 1e-2
}

# Generate data tensor
num_of_datasets = 10000
phases = np.empty((num_of_datasets, 256, 256, 1))
positions = np.empty((num_of_datasets, 4))

for i in range(num_of_datasets):
    generate_data(i)

data = h5py.File('../data/data.hdf5', 'w')
data.create_dataset('phases', data=phases)
data.create_dataset('positions', data=positions)
data.close()
