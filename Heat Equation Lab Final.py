# # -*- coding: utf-8 -*-
# """
# Created on Wed Apr 19 15:04:54 2023

# @author: ghitabelaid19
# """

from pylab import *
import matplotlib.pyplot as plt


ROD_LENGTH = 1
TIME = 1
beta = 1

def initialize_stiffness_matrix(num_space_steps, sigma):
    stiffness_mat = np.mat(np.zeros((num_space_steps - 1, num_space_steps - 1)))
    
    for i in range (0, num_space_steps - 1):
        stiffness_mat[i,i] = 1 - 2 * sigma
    for i in range (0, num_space_steps - 2):
        stiffness_mat[i, i+1] = sigma
        stiffness_mat[i+1, i] = sigma
        
    return stiffness_mat


def homogenous(delta_x, delta_t):
    num_space_steps = int(ROD_LENGTH / delta_x)
    num_time_steps = len(np.arange(0, TIME, delta_t))
    sigma = (beta * delta_t)/((delta_x)**2)
    
    # Initialize the stiffness matrix
    A = initialize_stiffness_matrix(num_space_steps, sigma)
    
    # Initialize initial condition vector
    U_t0 = np.zeros(((num_space_steps-1), 1))
    for i in range (num_space_steps-1):
        U_t0[i,0] = pow (sin(2* pi * (delta_x * (i + 1))), 2)
    
    # calculations
    U = [U_t0]
    U_t_prev = U_t0
    for i in range(1, num_time_steps):
        U_t_cur = A.dot(U_t_prev)
        U.append(U_t_cur)
        U_t_prev = U_t_cur
            
    for i in range(len(U)):
        U[i] = U[i].T
    
    u_array = np.stack(U, axis=0)
    
    bc_vector = np.zeros((len(u_array), 1))
    u_array = np.append(u_array, bc_vector, axis=1)
    u_array = np.insert(u_array, 0, 0, axis = 1)
    
    return u_array
    

def non_homogenous(delta_x, delta_t):
    num_space_steps = int(ROD_LENGTH / delta_x)
    num_time_steps = len(np.arange(0, TIME, delta_t))
    sigma = (beta * delta_t)/((delta_x)**2)

    # Initialize the stiffness matrix
    A = initialize_stiffness_matrix(num_space_steps, sigma)

    # Initialize initial condition vector
    U_t0 = np.zeros(((num_space_steps-1), 1))

    # calculations
    U = [U_t0]
    U_t_prev = U_t0

    bc_vector = np.zeros((num_space_steps - 1, 1))
    bc_vector[0] = 20
    bc_vector[-1] = 50

    for i in range(1, num_time_steps):
        U_t_cur = A.dot(U_t_prev) + sigma * bc_vector
        U.append(U_t_cur)
        U_t_prev = U_t_cur
            
    for i in range(len(U)):
        U[i] = U[i].T


    u_array = np.stack(U, axis=0)
    right_bound = np.full((len(u_array), 1), 50)
    u_array = np.append(u_array, right_bound, axis=1)
    u_array = np.insert(u_array, 0, 20, axis = 1)
    
    return u_array

def heatmap(delta_x, delta_t, z):
    u = np.asarray(z)
    x_range = np.arange(0, ROD_LENGTH + delta_x, delta_x)
    t_range = np.arange(0, TIME, delta_t)
    
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x_range, t_range, u, shading='nearest', cmap='hot', vmin=0)
    ax.set_title(f"delta_x = {delta_x}, delta_t = {delta_t}")
    fig.colorbar(c, ax=ax)
    plt.show()
    
def threed_map(delta_x, delta_t, z):
    x_range = np.arange(0, ROD_LENGTH + delta_x, delta_x)
    t_range = np.arange(0, TIME, delta_t)
    x, t = np.meshgrid(x_range, t_range)
    
    fig, axs = plt.subplots()
    ax = plt.subplot(projection = '3d')
    ax.set_title(f"delta_x = {delta_x}, delta_t = {delta_t}")
    surf = ax.plot_surface(x, t, z, cmap='hot')
    fig.colorbar(surf, ax=ax)
    
    plt.show()
    

def main():
    delta_x = .01
    delta_t = .004

    x_range = np.arange(0, ROD_LENGTH + delta_x, delta_x)
    t_range = np.arange(0, TIME, delta_t)
    
    # problem 1
    u = homogenous(delta_x, delta_t)
    # problem 2
    u = non_homogenous(delta_x, delta_t)
    
    # plot
    threed_map(delta_x, delta_t, u)
    heatmap(delta_x, delta_t, u)
    
main()


