from pygpe.shared.grid import Grid
import numpy as np


def generate_positions(grid: Grid, num_vortices: int, threshold: float) -> iter:
    """Generates and returns a list of positions that are separated by at least
    `threshold`.
    """
    max_iter = 10000
    vortex_positions = []

    iterations = 0
    while len(vortex_positions) < num_vortices:
        if iterations > max_iter:
            print(f"WARNING: Number of iterations exceeded maximum, "
                  f"returning with only {len(vortex_positions)} positions\n")
            return vortex_positions

        position = np.random.uniform(-grid.length_x / 3, grid.length_x / 3), \
                   np.random.uniform(-grid.length_y / 3, grid.length_y / 3)

        if _position_sufficiently_far(position, vortex_positions, threshold):
            vortex_positions.append(position)

        iterations += 1

    return vortex_positions


def _position_sufficiently_far(position: tuple, accepted_positions: list[tuple], threshold: float) -> bool:
    """Tests that the given `position` is at least `threshold` away from all the positions
    currently in `accepted_positions`.
    """
    # Special case where accepted_positions is empty
    if not accepted_positions:
        return True

    for accepted_pos in accepted_positions:
        if abs(position[0] - accepted_pos[0]) > threshold:
            if abs(position[1] - accepted_pos[1]) > threshold:
                return True
    return False


def _heaviside(array: np.ndarray) -> np.ndarray:
    """Computes the heaviside function on a given array and returns the result."""
    return np.where(array < 0, np.zeros(array.shape), np.ones(array.shape))


def vortex_phase_profile(grid: Grid, positions: list) -> np.ndarray:
    """Constructs a 2D phase profile consisting of 2pi phase windings.
    This phase can be applied to a wavefunction to generate different types of vortices.

    :param grid: The 2D grid of the system.
    :type grid: :class:`Grid`
    :param positions: List of vortex positions.
    :type positions: list
    """
    vortex_positions_iter = iter(positions)

    phase = np.zeros((grid.num_points_x, grid.num_points_y), dtype='float32')

    for _ in range(len(positions) // 2):
        phase_temp = np.zeros((grid.num_points_x, grid.num_points_y), dtype='float32')
        x_pos_minus, y_pos_minus = next(vortex_positions_iter)  # Negative circulation vortex
        x_pos_plus, y_pos_plus = next(vortex_positions_iter)  # Positive circulation vortex

        # Aux variables
        y_minus = 2 * np.pi / grid.length_y * (grid.y_mesh - y_pos_minus)
        x_minus = 2 * np.pi / grid.length_x * (grid.x_mesh - x_pos_minus)
        y_plus = 2 * np.pi / grid.length_y * (grid.y_mesh - y_pos_plus)
        x_plus = 2 * np.pi / grid.length_x * (grid.x_mesh - x_pos_plus)

        heaviside_x_plus = _heaviside(x_plus)
        heaviside_x_minus = _heaviside(x_minus)

        for nn in np.arange(-5, 6):
            phase_temp += np.arctan(np.tanh((y_minus + 2 * np.pi * nn) / 2) * np.tan((x_minus - np.pi) / 2)) \
                          - np.arctan(np.tanh((y_plus + 2 * np.pi * nn) / 2) * np.tan((x_plus - np.pi) / 2)) \
                          + np.pi * (heaviside_x_plus - heaviside_x_minus)
        phase_temp -= 2 * np.pi * (grid.y_mesh - grid.y_mesh.min()) \
                      * (x_pos_plus - x_pos_minus) / (grid.length_y * grid.length_x)
        phase += phase_temp

    return phase
