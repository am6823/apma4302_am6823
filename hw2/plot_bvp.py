import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt

def read_hdf5_vec(filename, vec_name):
    """
    Read PETSc HDF5 viewer output and convert to numpy arrays.
    
    Parameters:
    filename: str - path to the HDF5 file
    vec_name: str - name of the vector to read  
    
    Returns:
    numpy array containing the data
    """
    # Create a viewer for reading HDF5 files
    viewer = PETSc.Viewer().createHDF5(filename, 'r')   
    
    # Create a Vec to load the data
    vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    vec.setName(vec_name)
    vec.load(viewer)
   
    # Convert to numpy array
    array = vec.getArray()
    
    # Clean up
    vec.destroy()
    viewer.destroy()
    
    return array.copy()


def plot_bvp_solution(x, u_numeric, u_exact):
    """
    Plot the numerical and exact solutions of the BVP.
    
    Parameters:
    x: numpy array - grid points
    u_numeric: numpy array - numerical solution
    u_exact: numpy array - exact solution
    """
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(x, u_numeric, 'b-', label='Numerical Solution', linewidth=2)
    ax1.plot(x, u_exact, 'r--', label='Exact Solution', linewidth=2)
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('u(x)', fontsize=14)
    ax1.set_title('BVP Numerical vs Exact Solution', fontsize=16)
    ax1.legend(fontsize=12)
    ax1 .grid(True)

    ax2.plot(x, u_numeric - u_exact, 'g--', label='Error', linewidth=1)
    ax2.set_ylabel('Error', fontsize=14)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.set_ylim(-np.max(np.abs(u_numeric - u_exact)) * 3., np.max(np.abs(u_numeric - u_exact)) * 3.)

    plt.show()



import subprocess

def run_bvp(m, k, gamma=0.0, c=0.0, P=2):
    """
    Runs the PETSc BVP solver for given parameters.
    """
    cmd = [
        "mpiexec", "-np", str(P), "./bvp",
        "-bvp_m", str(m),
        "-bvp_gamma", str(gamma),
        "-bvp_k", str(k),
        "-bvp_c", str(c),
        "-bvp_write_hdf5"
    ]

    subprocess.run(cmd, check=True)


def compute_l2_error(u, u_exact):
    return np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)


def convergence_study():
    gamma = 0.0
    c = 0.0
    ks = [1, 5, 10]
    ms = [40, 80, 160, 320, 640, 1280]

    plt.figure()

    for k in ks:
        errors = []
        hs = []

        for m in ms:
            run_bvp(m, k, gamma=gamma, c=c)

            # Read from HDF5
            u = read_hdf5_vec("bvp_solution.h5", "u")
            u_exact = read_hdf5_vec("bvp_solution.h5", "uexact")

            err = compute_l2_error(u, u_exact)
            h = 1.0 / m

            errors.append(err)
            hs.append(h)

            print(f"k={k}, m={m}, h={h:.5e}, error={err:.5e}")

        hs = np.array(hs)
        errors = np.array(errors)

        # Fit order p from log-log slope
        p, _ = np.polyfit(np.log(hs), np.log(errors), 1)
        print(f"Estimated order for k={k}: p ≈ {p:.3f}")

        plt.loglog(hs, errors, marker='o', label=f"k={k}, p≈{p:.2f}")

    plt.xlabel("h = 1/m")
    plt.ylabel("Relative L2 Error")
    plt.title("Convergence Study (gamma = 0)")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


if __name__ == "__main__":
    plot_bvp_solution(np.linspace(0, 1, 100), np.sin(np.pi * np.linspace(0, 1, 100)), np.sin(np.pi * np.linspace(0, 1, 100)))
    convergence_study()