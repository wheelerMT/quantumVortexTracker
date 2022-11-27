import pygpe.scalar as gpe
from pygpe.shared.vortices import vortex_phase_profile
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def generate_image(iteration: int):
    """Generates grayscale images containing a random amount of vortices.

    :param iteration: Iteration number.
    """
    psi = gpe.Wavefunction(grid)
    psi.set_wavefunction(cp.ones(grid_points, dtype='complex128'))
    phase = vortex_phase_profile(grid, np.random.choice(vortex_numbers), 2)

    psi.apply_phase(cp.asarray(phase))
    psi.fft()
    for _ in range(params["nt"]):
        gpe.step_wavefunction(psi, params)
    psi.ifft()
    psi.add_noise(0, 1e-2)

    plt.imshow(cp.asnumpy(psi.density()), cmap='gray')
    plt.axis('off')
    plt.savefig(f'../data/test_{iteration}.png', bbox_inches='tight')


grid_points = (128, 128)
grid_spacing = (0.5, 0.5)
grid = gpe.Grid(grid_points, grid_spacing)
params = {
    "g": 1,
    "trap": 0,
    "nt": 100,
    "dt": -1j * 1e-2
}
vortex_numbers = [num for num in range(2, 22, 2)]

for i in range(10):
    generate_image(i)
