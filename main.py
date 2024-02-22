import numpy as np
from scipy.fftpack import fftfreq
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
#import h5py
def fraction_finder(positions_x, positions_y, x_grid, y_grid, dx, dy):
    x_frac = (positions_x - np.sum(x_grid[0])) / dx
    y_frac = (positions_y - np.sum(y_grid[0])) / dy

    return x_frac, y_frac


def periodic_particles(positions_x, positions_y, length_domain_x, length_domain_y):

    # NumPy implementation
    # Determine indices of particles which have gone outside the domain
    # through right boundary
    outside_domain_right_x = np.where(positions_x >= length_domain_x)
    outside_domain_top_y = np.where(positions_y >= length_domain_y)

    # Determine indices of particles which have gone outside the domain
    # through left boundary
    outside_domain_left_x = np.where(positions_x < 0)
    outside_domain_bottom_y = np.where(positions_y < 0)

    if len(outside_domain_right_x[0]) > 0:
        # Apply periodic boundary conditions
        positions_x[outside_domain_right_x] -= length_domain_x

    if len(outside_domain_top_y[0]) > 0:
        # Apply periodic boundary conditions
        positions_y[outside_domain_top_y] -= length_domain_y

    if len(outside_domain_left_x[0]) > 0:
        # Apply periodic boundary conditions
        positions_x[outside_domain_left_x] += length_domain_x

    if len(outside_domain_bottom_y[0]) > 0:
        # Apply periodic boundary conditions
        positions_y[outside_domain_bottom_y] += length_domain_y

    return positions_x, positions_y


def periodic_ghost(field, ghost_cells):
    len_y, len_x = field.shape  # Get the dimensions of the field array

    # Apply periodic boundary conditions
    field[0:ghost_cells, :] = field[len_y - 2 * ghost_cells:len_y - ghost_cells, :]
    field[:, 0:ghost_cells] = field[:, len_x - 2 * ghost_cells:len_x - ghost_cells]
    field[len_y - ghost_cells:len_y, :] = field[ghost_cells + 1:2 * ghost_cells + 1, :]
    field[:, len_x - ghost_cells:len_x] = field[:, ghost_cells + 1:2 * ghost_cells + 1]

    return field


def charge_b1_depositor(charge_electron, positions_x, positions_y, x_grid, y_grid, ghost_cells, length_domain_x,
                        length_domain_y):
    number_of_particles = len(positions_x)

    x_charge_zone = np.zeros(4 * number_of_particles, dtype=np.uint32)
    y_charge_zone = np.zeros(4 * number_of_particles, dtype=np.uint32)

    # calculating the number of grid cells
    nx = x_grid.size - 1 - 2 * ghost_cells  # number of zones
    ny = y_grid.size - 1 - 2 * ghost_cells  # number of zones

    dx = length_domain_x / nx
    dy = length_domain_y / ny

    # Determining the left(x) and bottom (y) indices of the left bottom corner grid node of
    # the grid cell containing the particle
    x_zone = np.floor(np.abs(positions_x - np.sum(x_grid[0])) / dx).astype(np.uint32)
    y_zone = np.floor(np.abs(positions_y - np.sum(y_grid[0])) / dy).astype(np.uint32)

    x_zone_plus = x_zone + 1
    y_zone_plus = y_zone + 1

    # Calculating the fractions needed for calculating the weights
    dy_by_delta_y = (1 / dy) * (positions_y - y_grid[y_zone])
    dy_by_delta_y_complement = 1 - dy_by_delta_y

    dx_by_delta_x = (1 / dx) * (positions_x - x_grid[x_zone])
    dx_by_delta_x_complement = 1 - dx_by_delta_x

    # Calculating the weights at all corners
    # Order of corners is available on the main thesis document
    # order -----bottom right --->bottom left---->top left-----> top right
    weight_corner1 = dy_by_delta_y_complement * dx_by_delta_x_complement
    weight_corner2 = dy_by_delta_y * dx_by_delta_x_complement
    weight_corner3 = dy_by_delta_y * dx_by_delta_x
    weight_corner4 = dy_by_delta_y_complement * dx_by_delta_x

    charge_by_dxdy = charge_electron / (dx * dy)

    corner1_charge = weight_corner1 * charge_by_dxdy
    corner2_charge = weight_corner2 * charge_by_dxdy
    corner3_charge = weight_corner3 * charge_by_dxdy
    corner4_charge = weight_corner4 * charge_by_dxdy

    # Concatenating the all the weights for all 4 corners into one vector all_corners_weighted_charge
    all_corners_weighted_charge = np.concatenate((corner1_charge, corner2_charge, corner3_charge, corner4_charge))

    # concatenating the x indices into x_charge_zone
    x_charge_zone[0 * number_of_particles:1 * number_of_particles] = x_zone
    x_charge_zone[1 * number_of_particles:2 * number_of_particles] = x_zone
    x_charge_zone[2 * number_of_particles:3 * number_of_particles] = x_zone_plus
    x_charge_zone[3 * number_of_particles:4 * number_of_particles] = x_zone_plus

    # concatenating the x indices into x_charge_zone
    y_charge_zone[0 * number_of_particles:1 * number_of_particles] = y_zone
    y_charge_zone[1 * number_of_particles:2 * number_of_particles] = y_zone_plus
    y_charge_zone[2 * number_of_particles:3 * number_of_particles] = y_zone_plus
    y_charge_zone[3 * number_of_particles:4 * number_of_particles] = y_zone

    return x_charge_zone, y_charge_zone, all_corners_weighted_charge


def cloud_charge_deposition(charge_electron, number_of_electrons, positions_x, positions_y, x_grid, y_grid,
                            shape_function, ghost_cells, length_domain_x, length_domain_y, dx, dy):

    elements = x_grid.size * y_grid.size

    rho_x_indices, rho_y_indices, rho_values_at_these_indices = shape_function(charge_electron, positions_x,
                                                                               positions_y, x_grid, y_grid, ghost_cells,
                                                                               length_domain_x, length_domain_y)

    input_indices = rho_x_indices * y_grid.size + rho_y_indices

    rho, _ = np.histogram(input_indices, bins=elements, range=(0, elements), weights=rho_values_at_these_indices)

    rho = rho.reshape((y_grid.size, x_grid.size))

    # Periodic BC's for charge deposition
    rho[ghost_cells, :] += rho[-1 - ghost_cells, :]
    rho[-1 - ghost_cells, :] = rho[ghost_cells, :].copy()
    rho[:, ghost_cells] += rho[:, -1 - ghost_cells]
    rho[:, -1 - ghost_cells] = rho[:, ghost_cells].copy()

    # Apply periodic ghost
    rho = periodic_ghost(rho, ghost_cells)

    return rho


def norm_background_ions(rho_electrons, number_of_electrons, w_p, charge_electron):
    A = 1 / (number_of_electrons * w_p)
    rho_electrons_normalized = A * rho_electrons

    # Adding background ion density, and ensuring charge neutrality
    rho_normalized = rho_electrons_normalized - charge_electron

    return rho_normalized

def poisson_solver(rho, dx, dy=None):
    # Calculate wavenumbers
    k_x = fftfreq(rho.shape[1], dx)
    k_y = fftfreq(rho.shape[0], dy) if dy is not None else None

    # Compute Fourier transform of charge density
    rho_hat = fft2(rho)

    # Compute potential in Fourier space
    kx_grid, ky_grid = np.meshgrid(k_x, k_y, indexing='ij')
    kx_grid_sq = kx_grid ** 2
    ky_grid_sq = ky_grid ** 2
    k_sq = kx_grid_sq + ky_grid_sq

    # Avoid division by zero at zero frequency
    k_sq[0, 0] = 1.0

    potential_hat = rho_hat.T / (4 * np.pi ** 2 * k_sq)

    # Compute electric field components in Fourier space
    E_x_hat = -1j * 2 * np.pi * kx_grid * potential_hat
    E_y_hat = -1j * 2 * np.pi * ky_grid * potential_hat if dy is not None else None

    # Compute inverse Fourier transform to get electric field in real space
    E_x = np.real(ifft2(E_x_hat))
    E_y = np.real(ifft2(E_y_hat)) if dy is not None else None

    return E_x, E_y

# Example usage:
# Define charge density rho and step sizes dx, dy
# rho = ...  # Define your charge density array
# dx = ...   # Define your step size in the x-direction
# dy = ...   # Define your step size in the y-direction (if applicable)
# E_x, E_y = poisson_solver(rho, dx, dy)

def set_up_perturbation(positions_x, positions_y, number_particles, N_divisions_x, N_divisions_y, amplitude, k_x, k_y, length_domain_x, length_domain_y, dx, dy):
    positions_x = length_domain_x * np.random.rand(number_particles)
    positions_y = length_domain_y * np.random.rand(number_particles)
    particles_till_x_i = 0
    for j in range(N_divisions_y):
        for i in range(N_divisions_x):
            average_particles_x_i_to_i_plus_one = (number_particles / ((length_domain_x * length_domain_y) / (dx * dy)))
            temp_amplitude = amplitude * np.cos((k_x * (i + 0.5) * dx / length_domain_x) + (k_y * (j + 0.5) * dy / length_domain_y))
            number_particles_x_i_to_i_plus_one = int(average_particles_x_i_to_i_plus_one * (1 + temp_amplitude))
            positions_x[particles_till_x_i : particles_till_x_i + number_particles_x_i_to_i_plus_one] = i * dx + dx * np.random.rand(number_particles_x_i_to_i_plus_one)
            positions_y[particles_till_x_i : particles_till_x_i + number_particles_x_i_to_i_plus_one] = j * dy + dy * np.random.rand(number_particles_x_i_to_i_plus_one)
            particles_till_x_i += number_particles_x_i_to_i_plus_one
    return positions_x, positions_y

def umeda_b1_deposition(charge_electron, positions_x, positions_y, velocity_x, velocity_y,x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y, dt):
    # Determine grid properties
    nx = x_grid.size - 1 - 2 * ghost_cells
    ny = y_grid.size - 1 - 2 * ghost_cells
    dx = length_domain_x / nx
    dy = length_domain_y / ny

    # Compute initial and final particle positions
    x_1 = positions_x.astype(float)
    x_2 = (positions_x + velocity_x * dt).astype(float)
    y_1 = positions_y.astype(float)
    y_2 = (positions_y + velocity_y * dt).astype(float)

    # Calculate indices of left corners of cells containing particles
    i_1 = np.floor(((np.abs(x_1 - np.sum(x_grid[0])))/dx) - ghost_cells).astype(int)
    j_1 = np.floor(((np.abs(y_1 - np.sum(y_grid[0])))/dy) - ghost_cells).astype(int)
    i_2 = np.floor(((np.abs(x_2 - np.sum(x_grid[0])))/dx) - ghost_cells).astype(int)
    j_2 = np.floor(((np.abs(y_2 - np.sum(y_grid[0])))/dy) - ghost_cells).astype(int)

    # Compute relay points
    x_r = np.minimum(dx + np.minimum(np.maximum(dx / 2, (x_1 + x_2) / 2), x_2), x_1)
    y_r = np.minimum(dy + np.minimum(np.maximum(dy / 2, (y_1 + y_2) / 2), y_2), y_1)

    # Compute fluxes and weights
    F_x_1 = charge_electron * (x_r - x_1) / dt
    F_x_2 = charge_electron * (x_2 - x_r) / dt
    F_y_1 = charge_electron * (y_r - y_1) / dt
    F_y_2 = charge_electron * (y_2 - y_r) / dt

    W_x_1 = (x_1 + x_r) / (2 * dx) - i_1
    W_x_2 = (x_2 + x_r) / (2 * dx) - i_2
    W_y_1 = (y_1 + y_r) / (2 * dy) - j_1
    W_y_2 = (y_2 + y_r) / (2 * dy) - j_2

    # Compute charge densities
    J_x_1_1 = (1 / (dx * dy)) * (F_x_1 * (1 - W_y_1))
    J_x_1_2 = (1 / (dx * dy)) * (F_x_1 * (W_y_1))
    J_x_2_1 = (1 / (dx * dy)) * (F_x_2 * (1 - W_y_2))
    J_x_2_2 = (1 / (dx * dy)) * (F_x_2 * (W_y_2))
    J_y_1_1 = (1 / (dx * dy)) * (F_y_1 * (1 - W_x_1))
    J_y_1_2 = (1 / (dx * dy)) * (F_y_1 * (W_x_1))
    J_y_2_1 = (1 / (dx * dy)) * (F_y_2 * (1 - W_x_2))
    J_y_2_2 = (1 / (dx * dy)) * (F_y_2 * (W_x_2))

    # Compute indices for deposition
    Jx_x_indices = np.concatenate([i_1 + ghost_cells, i_1 + ghost_cells, i_2 + ghost_cells, i_2 + ghost_cells])
    Jx_y_indices = np.concatenate([j_1 + ghost_cells, j_1 + 1 + ghost_cells, j_2 + ghost_cells, j_2 + 1 + ghost_cells])
    Jy_x_indices = np.concatenate([i_1 + ghost_cells, i_1 + 1 + ghost_cells, i_2 + ghost_cells, i_2 + 1 + ghost_cells])
    Jy_y_indices = np.concatenate([j_1 + ghost_cells, j_1 + ghost_cells, j_2 + ghost_cells, j_2 + ghost_cells])

    # Compute values for deposition
    Jx_values_at_these_indices = np.concatenate([J_x_1_1, J_x_1_2, J_x_2_1, J_x_2_2])
    Jy_values_at_these_indices = np.concatenate([J_y_1_1, J_y_1_2, J_y_2_1, J_y_2_2])

    return Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices, Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices

def umeda_2003(charge_electron, number_of_electrons, positions_x, positions_y,velocities_x, velocities_y, x_grid, y_grid, ghost_cells,length_domain_x, length_domain_y, dx, dy, dt):
    elements = x_grid.size * y_grid.size
    # Call Umeda_b1_deposition to get current deposition values
    Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices = umeda_b1_deposition(charge_electron, positions_x, positions_y,velocities_x, velocities_y, x_grid, y_grid,ghost_cells, length_domain_x, length_domain_y, dt)

    # Current deposition using numpy's histogram
    input_indices_Jx = Jx_x_indices * y_grid.size + Jx_y_indices
    input_indices_Jy = Jy_x_indices * y_grid.size + Jy_y_indices

    # Computing Jx_Yee
    Jx_Yee, _ = np.histogram(input_indices_Jx, bins=elements, range=(0, elements),weights=Jx_values_at_these_indices)
    Jx_Yee = np.reshape(Jx_Yee, (y_grid.size, x_grid.size))

    # Computing Jy_Yee
    Jy_Yee, _ = np.histogram(input_indices_Jy, bins=elements, range=(0, elements),weights=Jy_values_at_these_indices)
    Jy_Yee = np.reshape(Jy_Yee, (y_grid.size, x_grid.size))

    return Jx_Yee, Jy_Yee

def current_norm_BC_Jx(Jx_Yee, number_of_electrons, w_p, ghost_cells):
    len_x, len_y = Jx_Yee.shape
    # Normalizing the currents to be deposited
    A = 1 / (number_of_electrons * w_p)
    Jx_norm_Yee = A * Jx_Yee.copy()

    # Assigning the current density to the boundary points for periodic boundary conditions
    Jx_norm_Yee[:, ghost_cells] += Jx_norm_Yee[:, -1 - ghost_cells]
    Jx_norm_Yee[:, -1 - ghost_cells] = Jx_norm_Yee[:, ghost_cells].copy()

    Jx_norm_Yee[:, -2 - ghost_cells] += Jx_norm_Yee[:, ghost_cells - 1]

    Jx_norm_Yee[:, ghost_cells + 1] += Jx_norm_Yee[:, -ghost_cells]

    # Assigning the current density to the boundary points in top and bottom rows along y direction
    Jx_norm_Yee[ghost_cells, :] += Jx_norm_Yee[ghost_cells, :]
    Jx_norm_Yee[-1 - ghost_cells, :] = Jx_norm_Yee[ghost_cells, :].copy()

    Jx_norm_Yee[ghost_cells + 1, :] += Jx_norm_Yee[ghost_cells + 1, :]
    Jx_norm_Yee[-2 - ghost_cells, :] += Jx_norm_Yee[-2 - ghost_cells, :]

    # Assigning ghost cell values
    #Jx_norm_Yee[:ghost_cells, :] += Jx_norm_Yee[-1 - ghost_cells:, :]
    Jx_norm_Yee[:1, :] += Jx_norm_Yee[-1:, :]
    Jx_norm_Yee[-ghost_cells:, :] += Jx_norm_Yee[:ghost_cells, :]
    # Check if Jx_norm_Yee has the same shape as Jx_Yee
    if Jx_norm_Yee.shape != Jx_Yee.shape:
    # Reshape Jx_norm_Yee to match the shape of Jx_Yee
      Jx_norm_Yee = Jx_norm_Yee.reshape(Jx_Yee.shape)
    return Jx_norm_Yee

def current_norm_BC_Jy(Jy_Yee, number_of_electrons, w_p, ghost_cells):
    len_x, len_y = Jy_Yee.shape
    # Normalizing the currents to be deposited
    A = 1 / (number_of_electrons * w_p)
    Jy_norm_Yee = A * Jy_Yee.copy()

    # Assigning the current density to the boundary points for periodic boundary conditions
    Jy_norm_Yee[ghost_cells, :] += Jy_norm_Yee[-1 - ghost_cells, :]
    Jy_norm_Yee[-1 - ghost_cells, :] = Jy_norm_Yee[ghost_cells, :].copy()

    Jy_norm_Yee[-2 - ghost_cells, :] += Jy_norm_Yee[ghost_cells - 1, :]

    Jy_norm_Yee[ghost_cells + 1, :] += Jy_norm_Yee[-ghost_cells, :]

    # Assigning the current density to the boundary points in left and right columns along x direction
    Jy_norm_Yee[:, ghost_cells] += Jy_norm_Yee[:, -1 - ghost_cells]
    Jy_norm_Yee[:, -1 - ghost_cells] = Jy_norm_Yee[:, ghost_cells].copy()

    Jy_norm_Yee[:, ghost_cells + 1] += Jy_norm_Yee[:, -ghost_cells]
    Jy_norm_Yee[:, -2 - ghost_cells] += Jy_norm_Yee[:, ghost_cells - 1]

    # Assigning ghost cell values
    #Jy_norm_Yee[:, :ghost_cells] += Jy_norm_Yee[:, -1 - ghost_cells:]
    Jy_norm_Yee[:1, :] += Jy_norm_Yee[-1:, :]
    Jy_norm_Yee[:, -ghost_cells:] += Jy_norm_Yee[:, :ghost_cells]

    return Jy_norm_Yee
'''
def fdtd(Ex, Ey, Bz, Lx, Ly, ghost_cells, Jx, Jy, dt):
    forward_row = np.array([1, -1, 0])
    forward_column = np.array([1, -1, 0])
    backward_row = np.array([0, 1, -1])
    backward_column = np.array([0, 1, -1])
    identity = np.array([0, 1, 0])

    x_number_of_points, y_number_of_points = Bz.shape

    Nx = x_number_of_points - 2 * ghost_cells - 1
    Ny = y_number_of_points - 2 * ghost_cells - 1

    Bz_local = Bz.copy()
    Ex_local = Ex.copy()
    Ey_local = Ey.copy()

    Bz_local = periodic_ghost(Bz_local, ghost_cells)
    Ex_local = periodic_ghost(Ex_local, ghost_cells)
    Ey_local = periodic_ghost(Ey_local, ghost_cells)

    dx = float(Lx / (Nx))
    dy = float(Ly / (Ny))

    dt_by_dx = dt / dx
    dt_by_dy = dt / dy
    # Reshape Bz_local to a 1-dimensional array
    #Bz_local = Bz_local.reshape(-1,1)
    #Bz_local = Bz_local.reshape(Bz.shape)
    #Bz_local = Bz_local.reshape(backward_row.shape)
    Ex_local += dt_by_dy * (np.convolve(Bz_local.flatten(), backward_row, mode='valid')) - Jx * dt
    Ey_local += -dt_by_dx * (np.convolve(Bz_local.flatten(), backward_column, mode='valid')) - Jy * dt

    Ex_local = periodic_ghost(Ex_local, ghost_cells)
    Ey_local = periodic_ghost(Ey_local, ghost_cells)

    Bz_local += -dt_by_dx * (np.convolve(Ey_local, forward_column, mode='valid')) + dt_by_dy * (np.convolve(Ex_local, forward_row, mode='valid'))
    Bz_local = periodic_ghost(Bz_local, ghost_cells)
    return Ex_local, Ey_local, Bz_local
'''
def fdtd(Ex, Ey, Bz, Lx, Ly, ghost_cells, Jx, Jy, dt):
    forward_row = np.array([1, -1, 0])
    forward_column = np.array([1, -1, 0])
    backward_row = np.array([0, 1, -1])
    backward_column = np.array([0, 1, -1])
    identity = np.array([0, 1, 0])

    x_number_of_points, y_number_of_points = Bz.shape

    Nx = x_number_of_points - 2 * ghost_cells - 1
    Ny = y_number_of_points - 2 * ghost_cells - 1

    Bz_local = Bz.copy()
    Ex_local = Ex.copy()
    Ey_local = Ey.copy()

    Bz_local = periodic_ghost(Bz_local, ghost_cells)
    Ex_local = periodic_ghost(Ex_local, ghost_cells)
    Ey_local = periodic_ghost(Ey_local, ghost_cells)

    dx = float(Lx / (Nx))
    dy = float(Ly / (Ny))

    dt_by_dx = dt / dx
    dt_by_dy = dt / dy

    # Debug: Check shapes of arrays before convolution
    print("Shapes before convolution:")
    print("Bz_local:", Bz_local.shape)
    print("Ex_local:", Ex_local.shape)
    print("Ey_local:", Ey_local.shape)

    # Perform convolution operations
    # Perform convolution operations for Ex_local
    conv_result_Ex = np.convolve(Bz_local.flatten(), backward_row, mode='same')  # Use 'same' mode
    conv_result_Ex = conv_result_Ex.reshape(Ex_local.shape)  # Reshape convolution result for Ex_local
    Ex_local += dt_by_dy * conv_result_Ex - Jx * dt

    # Perform convolution operations for Ey_local
    conv_result_Ey = np.convolve(Bz_local.flatten(), backward_column, mode='same')  # Use 'same' mode
    conv_result_Ey = conv_result_Ey.reshape(Ey_local.shape)  # Reshape convolution result for Ey_local
    Ey_local += -dt_by_dx * conv_result_Ey - Jy * dt

    # Debug: Check shapes after convolution
    print("Shapes after convolution:")
    print("Ex_local:", Ex_local.shape)
    print("Ey_local:", Ey_local.shape)

    # Perform convolution operations for Bz_local
    conv_result_Bz = np.convolve(Ey_local.flatten(), forward_column, mode='same') - np.convolve(Ex_local.flatten(),
                                                                                                forward_row,
                                                                                                mode='same')
    conv_result_Bz = conv_result_Bz.reshape(Bz_local.shape)  # Reshape convolution result for Bz_local
    Bz_local += -dt_by_dx * conv_result_Bz + dt_by_dy * conv_result_Bz

    # Debug: Check final shape before returning
    print("Final shape before return - Bz_local:", Bz_local.shape)

    Bz_local = periodic_ghost(Bz_local, ghost_cells)
    return Ex_local, Ey_local, Bz_local

# Helper function periodic_ghost() needs to be defined




def Boris(charge_electron, mass_electron, velocity_x, velocity_y, dt, Ex_particle, Ey_particle, Bz_particle):
    vel_x_minus = velocity_x + (charge_electron * Ex_particle * dt) / (2 * mass_electron)
    vel_y_minus = velocity_y + (charge_electron * Ey_particle * dt) / (2 * mass_electron)

    t_magz = (charge_electron * Bz_particle * dt) / (2 * mass_electron)

    vminus_cross_t_x = vel_y_minus * t_magz
    vminus_cross_t_y = -vel_x_minus * t_magz

    vel_dashx = vel_x_minus + vminus_cross_t_x
    vel_dashy = vel_y_minus + vminus_cross_t_y

    t_mag = np.sqrt(t_magz ** 2)

    s_z = (2 * t_magz) / (1 + np.abs(t_mag ** 2))

    vel_x_plus = vel_x_minus + (vel_dashy * s_z)
    vel_y_plus = vel_y_minus - (vel_dashx * s_z)

    velocity_x_new = vel_x_plus + (charge_electron * Ex_particle * dt) / (2 * mass_electron)
    velocity_y_new = vel_y_plus + (charge_electron * Ey_particle * dt) / (2 * mass_electron)

    return velocity_x_new, velocity_y_new

def interpolate_field(field, frac_y, frac_x):

    # Find the indices of the four nearest grid points
    ix = int(frac_x[0])
    iy = int(frac_y[0])

    # Calculate the weights for each grid point
    wx = frac_x - ix
    wy = frac_y - iy

    # Interpolate the field value
    interpolated_value = (1 - wx) * (1 - wy) * field[iy, ix] + \
                         wx * (1 - wy) * field[iy, ix + 1] + \
                         (1 - wx) * wy * field[iy + 1, ix] + \
                         wx * wy * field[iy + 1, ix + 1]

    return interpolated_value

w_p             = 1

# Macro Particle parameters
k_boltzmann     = 1
mass_electron   = 1 * w_p
tempertature    = 1
charge_electron = -10 * w_p
charge_ion      = +10 * w_p

# Setting the length of the domain
length_domain_x = 1
length_domain_y = 1

# Setting the number of ghost cells
ghost_cells  = 1
# Setting number of particle in the domain
number_of_electrons = 100 #6100000

# Initializing the positions and velocities of the particles
positions_x = length_domain_x * np.random.rand(number_of_electrons)
positions_y = length_domain_y * np.random.rand(number_of_electrons)

# setting the mean and standard deviation of the maxwell distribution
# Thermal/mean velocity of macro particles should correspond to
# that of individual electrons in the plasma
mu_x, sigma_x = 0, (k_boltzmann * tempertature / (mass_electron / w_p))
mu_y, sigma_y = 0, (k_boltzmann * tempertature / (mass_electron / w_p))

# Initializing the velocitites according to the maxwell distribution
velocity_x = np.random.normal(mu_x, sigma_x, number_of_electrons)
velocity_y = np.random.normal(mu_y, sigma_y, number_of_electrons)

# Divisions in x grid
divisions_domain_x = 100
divisions_domain_y = 2

# dx, dy is the distance between consecutive grid nodes along x and y
dx = (length_domain_x / divisions_domain_x)
dy = (length_domain_y / divisions_domain_y)

# initializing the x grid
x_grid = np.linspace(0 - ghost_cells * dx,length_domain_x + ghost_cells * dx,divisions_domain_x + 1 + 2 * ghost_cells,endpoint=True,dtype=np.double)
x_right = x_grid + dx/2


# initializing the y grid
y_grid = np.linspace(0 - ghost_cells * dy,length_domain_y + ghost_cells * dy,divisions_domain_y + 1 + 2 * ghost_cells,endpoint=True,dtype=np.double)
y_top = y_grid + dy/2

# Setting the amplitude for perturbation
N_divisions_x = divisions_domain_x
N_divisions_y = divisions_domain_y
amplitude_perturbed = 0.5
k_x = 2 * np.pi
k_y = 2 * np.pi  # Adjust the wave number for 2D perturbation

# Initializing the perturbation
positions_x, positions_y = set_up_perturbation(positions_x,positions_y,number_of_electrons,N_divisions_x,N_divisions_y,amplitude_perturbed,k_x,k_y,length_domain_x,length_domain_y,dx,dy)

# For 2D simulation:
velocity_y = np.zeros(number_of_electrons)  # Initialize velocity in the y-direction

# Converting to NumPy arrays
positions_x = np.array(positions_x)
positions_y = np.array(positions_y)
velocity_x = np.array(velocity_x)
velocity_y = np.array(velocity_y)
x_grid = np.array(x_grid)
y_grid = np.array(y_grid)
x_right = np.array(x_right)
y_top = np.array(y_top)

position_grid = np.linspace(0, 1, N_divisions_x)
number_electrons_in_bins, b = np.histogram(np.array(positions_x), bins=100, range=(0, length_domain_x))
number_density = number_electrons_in_bins / (number_of_electrons / divisions_domain_x)

plt.plot(position_grid, number_density)
plt.xlabel('Position')
plt.ylabel('Number Density')
plt.title('Initial Density Perturbation')
plt.show()

# Time parameters
start_time = 0
end_time   = 1
dt         = 0.002
time       = np.arange(start_time,end_time + dt,dt,dtype = np.double)

# Some variables for storing data
Ex_max       = np.zeros(len(time), dtype = np.double)
Ey_max       = np.zeros(len(time), dtype = np.double)

# Charge deposition using linear weighting scheme
rho_electrons  = cloud_charge_deposition(charge_electron,number_of_electrons,positions_x,positions_y,x_grid,y_grid,charge_b1_depositor,ghost_cells,length_domain_x,length_domain_y,dx,dy)
rho_initial    = norm_background_ions(rho_electrons, number_of_electrons, w_p, charge_electron)

plt.plot(np.array(rho_initial)[1, :])
plt.title('Initial Charge Density')
plt.xlabel('Grid Index')
plt.ylabel('Charge Density')
plt.show()
plt.clf()

Ex_initial_centered = np.zeros((y_grid.size, x_grid.size))
Ey_initial_centered = np.zeros((y_grid.size, x_grid.size))

rho_physical = rho_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells].copy()

Ex_temp, Ey_temp = poisson_solver(rho_physical, dx, dy)

Ex_initial_centered[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = Ex_temp.T  #copy()
Ey_initial_centered[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = Ey_temp.T  #copy()

Ex_initial_Yee = 0.5 * (Ex_initial_centered + np.roll(Ex_initial_centered, -1, axis=1))
Ex_initial_Yee = periodic_ghost(Ex_initial_Yee, ghost_cells)

Ey_initial_Yee = 0.5 * (Ey_initial_centered + np.roll(Ey_initial_centered, -1, axis=0))
Ey_initial_Yee = periodic_ghost(Ey_initial_Yee, ghost_cells)

# Obtain v at (t = 0.5dt) to implement the Verlet algorithm
positions_x_half = positions_x + velocity_x * dt / 2
positions_y_half = positions_y + velocity_y * dt / 2

# Periodic Boundary conditions for particles
positions_x_half, positions_y_half = periodic_particles(positions_x_half, positions_y_half, length_domain_x, length_domain_y)

# Finding interpolant fractions for the positions
fracs_Ex_x, fracs_Ex_y = fraction_finder(positions_x_half, positions_y_half, x_right, y_grid, dx, dy)
fracs_Ey_x, fracs_Ey_y = fraction_finder(positions_x_half, positions_y_half, x_grid, y_top, dx, dy)

# Interpolating the fields at each particle
Ex_particle = interpolate_field(Ex_initial_Yee, fracs_Ex_y, fracs_Ex_x)
Ey_particle = interpolate_field(Ey_initial_Yee, fracs_Ey_y, fracs_Ey_x)

# Updating the velocity using the interpolated Electric fields to find v at (t = 0.5dt)
velocity_x += (Ex_particle * charge_electron / mass_electron) * dt / 2
velocity_y += (Ey_particle * charge_electron / mass_electron) * dt / 2

Ex = Ex_initial_Yee.copy()
Ey = Ey_initial_Yee.copy()
Bz = np.zeros_like(Ey_initial_Yee)

plt.plot(np.array(Ex_initial_Yee)[1, :])
plt.show()
plt.clf()

for time_index in range(len(time)):
    if time_index % 25 == 0:
        print('Computing for time = ', time_index * dt)

    # Updating the positions of particles using the velocities (Verlet algorithm)
    # velocity at t = (n + 1/2) dt, positions_x at t = (n)dt, and positions_x_new
    # at t = (n+1)dt
    positions_x_new = positions_x + velocity_x * dt
    positions_y_new = positions_y + velocity_y * dt

    # Applying periodic boundary conditions for particles
    positions_x_new, positions_y_new = periodic_particles(positions_x_new, positions_y_new,length_domain_x, length_domain_y)

    # Computing the current densities on the Yee grid provided by Umeda's scheme
    Jx_Yee, Jy_Yee = umeda_2003(charge_electron,number_of_electrons,positions_x, positions_y,velocity_x, velocity_y,x_grid, y_grid,ghost_cells,length_domain_x, length_domain_y,dx, dy,dt)

    # Normalizing and evaluating the current densities on the centered grid
    Jx_norm_Yee = current_norm_BC_Jx(Jx_Yee, number_of_electrons, w_p,ghost_cells)
    Jy_norm_Yee = current_norm_BC_Jy(Jy_Yee, number_of_electrons, w_p,ghost_cells)

    # Evolving electric fields using currents
    Ex_updated, Ey_updated, Bz_updated = fdtd(Ex, Ey, Bz, length_domain_x,length_domain_y, ghost_cells,Jx_norm_Yee, Jy_norm_Yee,dt)
    # Forcing Bz to be zero
    Bz_updated[:, :] = 0

    # calculating the interpolation fractions needed for 2D interpolation
    fracs_Ex_x, fracs_Ex_y = fraction_finder(positions_x_new, positions_y_new,x_right, y_grid, dx, dy)
    fracs_Ey_x, fracs_Ey_y = fraction_finder(positions_x_new, positions_y_new,x_grid, y_top, dx, dy)
    fracs_Bz_x, fracs_Bz_y = fraction_finder(positions_x_new, positions_y_new,x_right, y_top, dx, dy)

    # Interpolating the fields at particle locations
    Ex_particle = interpolate_field(Ex_updated, fracs_Ex_y, fracs_Ex_x)
    Ey_particle = interpolate_field(Ey_updated, fracs_Ey_y, fracs_Ey_x)

    # Computing the interpolated magnetic field at t = (n+1)*dt to push v((n+1/2) * dt)
    Bz_particle = interpolate_field((Bz_updated + Bz) / 2, fracs_Bz_y, fracs_Bz_x)

    # Updating the velocity using the interpolated Electric fields
    velocity_x_new, velocity_y_new = Boris(charge_electron, mass_electron,velocity_x, velocity_y, dt,Ex_particle, Ey_particle, Bz_particle)

    # Saving the Electric fields for plotting
    Ex_max[time_index] = np.max(np.abs(Ex[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]))
    Ey_max[time_index] = np.max(np.abs(Ey[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]))

    # Saving the updated velocities for the next timestep
    velocity_x = velocity_x_new.copy()
    positions_x = positions_x_new.copy()
    velocity_y = velocity_y_new.copy()
    positions_y = positions_y_new.copy()
    Ex = Ex_updated.copy()
    Ey = Ey_updated.copy()
    '''
    h5f = h5py.File('PIC_2D_CD.h5', 'w')
    h5f.create_dataset('Ex_amp', data=(Ex_max))
    h5f.create_dataset('Ey_amp', data=(Ey_max))
    h5f.close()

    # Open the HDF5 file in read mode
    with h5py.File('CK_256.h5', 'r') as h5f:
        # Read the dataset 'max_E' into the variable Ex_max_CK
        Ex_max_CK = h5f['max_E'][:]
    # Create a time array with the same length as Ex_max_CK
    time_CK = np.linspace(0, time[-1], len(Ex_max_CK))
    '''
    # Create a time grid
    time_grid = np.linspace(0, time[-1], len(Ex_max))

    # Plot the maximum electric field amplitude for PIC and Cheng Knorr
    plt.plot(time_grid, Ex_max, label=r'$\mathrm{PIC}$')
    #plt.plot(time_CK, Ex_max_CK, label=r'$\mathrm{Cheng\ Knorr}$')

    # Add labels and legend
    plt.xlabel('$t$')
    plt.ylabel('$\mathrm{MAX}(|E_{x}|)$')
    plt.legend()

    # Display the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig('MaxE.png')

    # Clear the current figure to release memory
    plt.clf()

    # Plot the maximum electric field amplitude for PIC and Cheng Knorr using semilogy
    plt.semilogy(time_grid, Ex_max, label=r'$\mathrm{PIC}$')
    #plt.semilogy(time_CK, Ex_max_CK, label=r'$\mathrm{Cheng\ Knorr}$')

    # Add labels and legend
    plt.xlabel('$t$')
    plt.ylabel('$\mathrm{MAX}(|E_{x}|)$')
    plt.legend()

    # Display the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig('MaxE_semilogy.png')

    # Clear the current figure to release memory
    plt.clf()

    # Plot the maximum electric field amplitude for PIC and Cheng Knorr using loglog
    plt.loglog(time_grid, Ex_max, label=r'$\mathrm{PIC}$')
    #plt.loglog(time_CK, Ex_max_CK, label=r'$\mathrm{Cheng\ Knorr}$')

    # Add labels and legend
    plt.xlabel('$t$')
    plt.ylabel('$\mathrm{MAX}(|E_{x}|)$')
    plt.legend()

    # Display the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig('MaxE_loglog.png')

    # Clear the current figure to release memory
    plt.clf()
