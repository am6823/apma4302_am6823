from firedrake import *
import firedrake_ts
import os
import math
import argparse

# Parameters for Question 4

N = 128

parser = argparse.ArgumentParser()
parser.add_argument("--Ra", type=float, default=1.0e2)
args = parser.parse_args()

Ra_value = args.Ra
Ra = Constant(Ra_value)

A = Constant(0.1)

t0 = 0.0
tmax = 100000.0
dt = 1000.0

# Mesh and function spaces

mesh = UnitSquareMesh(N, N, quadrilateral=True)

V = FunctionSpace(mesh, "Lagrange", 1)
ME = MixedFunctionSpace([V, V, V], name=["temperature", "vorticity", "streamfunction"])
Vv = VectorFunctionSpace(mesh, "Lagrange", 1)

# Unknown and time derivative
u = Function(ME, name="solution")
udot = Function(ME, name="solution_dot")

T, omega, psi = split(u)
Tdot, omegadot, psidot = split(udot)

# Test functions
q, eta, phi = TestFunctions(ME)

# Coordinates
x, y = SpatialCoordinate(mesh)

# Initial condition
# T(x,y,0) = (1-y) + A cos(pi x)

T0 = (1.0 - y) + A * cos(pi * x)

u.subfunctions[0].interpolate(T0)
u.subfunctions[0].rename("temperature")

u.subfunctions[1].assign(0.0)
u.subfunctions[1].rename("vorticity")

u.subfunctions[2].assign(0.0)
u.subfunctions[2].rename("streamfunction")

# Useful: initialize omega and psi consistently
# with the initial temperature field.
# This solves:
# -Delta omega = dT/dx
# -Delta psi = omega

W = MixedFunctionSpace([V, V], name=["vorticity_init", "streamfunction_init"])
w = Function(W)

omega_i, psi_i = split(w)
eta_i, phi_i = TestFunctions(W)

T_initial = u.subfunctions[0]

F_init = (
    inner(grad(omega_i), grad(eta_i)) * dx
    - eta_i * T_initial.dx(0) * dx
    + inner(grad(psi_i), grad(phi_i)) * dx
    - phi_i * omega_i * dx
)

bcs_init = [
    DirichletBC(W.sub(0), 0.0, "on_boundary"),
    DirichletBC(W.sub(1), 0.0, "on_boundary"),
]

init_params = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij",
}

solve(F_init == 0, w, bcs=bcs_init, solver_parameters=init_params)

u.subfunctions[1].assign(w.subfunctions[0])
u.subfunctions[2].assign(w.subfunctions[1])

# Weak form of the coupled DAE system

# Velocity from streamfunction
v_adv = curl(psi)

# Temperature equation:
# T_t + v . grad(T) = (1/Ra) Delta T
F_T = (
    Tdot * q * dx
    + dot(v_adv, grad(T)) * q * dx
    + (1.0 / Ra) * inner(grad(T), grad(q)) * dx
)

# Vorticity equation:
# -Delta omega = dT/dx
F_omega = (
    inner(grad(omega), grad(eta)) * dx
    - eta * T.dx(0) * dx
)

# Streamfunction equation:
# -Delta psi = omega
F_psi = (
    inner(grad(psi), grad(phi)) * dx
    - phi * omega * dx
)

F = F_T + F_omega + F_psi

# Boundary conditions
# For UnitSquareMesh:
# 1: left, 2: right, 3: bottom, 4: top

bcs = [
    # Temperature Dirichlet BCs
    DirichletBC(ME.sub(0), 1.0, 3),  # y = 0
    DirichletBC(ME.sub(0), 0.0, 4),  # y = 1

    # Vorticity and streamfunction
    DirichletBC(ME.sub(1), 0.0, "on_boundary"),
    DirichletBC(ME.sub(2), 0.0, "on_boundary"),
]


# Diagnostics: velocity and Nusselt number

def compute_diagnostics():
    T_h = u.subfunctions[0]
    omega_h = u.subfunctions[1]
    psi_h = u.subfunctions[2]

    v_h = Function(Vv, name="velocity")
    v_h.interpolate(curl(psi_h))

    # For the conductive solution T = 1 - y,
    # -int_top dT/dy dx = 1.
    dTdy_top = assemble(T_h.dx(1) * ds(4))
    dTdy_bottom = assemble(T_h.dx(1) * ds(3))
    Tmean_bottom = assemble(T_h * ds(3))

    

    Nu_top = -float(dTdy_top)
    Nu_bottom = -float(dTdy_bottom)
    Nu = Nu_top/Tmean_bottom

    T_error = Function(V, name="T_minus_conductive")
    T_error.interpolate(T_h - (1.0 - y))

    T_error_L2 = math.sqrt(float(assemble(T_error * T_error * dx)))
    velocity_L2 = math.sqrt(float(assemble(dot(v_h, v_h) * dx)))

    return v_h, Nu_top, Nu_bottom, Nu, T_error_L2, velocity_L2


# Output

outfile = VTKFile(f"result/q4_convection_Ra{Ra_value:.0e}.pvd")

def write_output(time_value):
    T_h = u.subfunctions[0]
    omega_h = u.subfunctions[1]
    psi_h = u.subfunctions[2]

    T_h.rename("temperature")
    omega_h.rename("vorticity")
    psi_h.rename("streamfunction")

    v_h, Nu_top, Nu_bottom, Nu, T_error_L2, velocity_L2 = compute_diagnostics()

    outfile.write(T_h, omega_h, psi_h, v_h, time=time_value)

    print(
        f"t = {time_value:.4e}, "
        f"Nu_top = {Nu_top:.8f}, "
        f"Nu_bottom = {Nu_bottom:.8f}, "
        f"Nu = {Nu:.8f}, "
        f"||T-(1-y)||_L2 = {T_error_L2:.4e}, "
        f"||v||_L2 = {velocity_L2:.4e}"
    )


# Initial output
write_output(t0)

# PETSc TS / firedrake-ts setup

time = Constant(t0)

problem = firedrake_ts.DAEProblem(
    F,
    u,
    udot,
    (t0, tmax),
    bcs=bcs,
    time=time,
)

solver_params = {
    # Time integration
    "ts_type": "bdf",
    "ts_bdf_order": 2,
    "ts_time_step": dt,
    #"ts_adapt_type": "none",
    "ts_adapt_type": "basic",
    "ts_adapt_dt_max": 500.0,
    "ts_adapt_dt_min": .1,
    "ts_exact_final_time": "matchstep",
    "ts_max_steps": 200000,
    "ts_monitor": None,

    # Nonlinear solver
    "snes_type": "newtonls",
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-10,
    "snes_max_it": 20,

    # Linear solver: monolithic direct solve with MUMPS
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij",
}


def monitor(ts, step, t, x):
    if step % 10 == 0:
        write_output(t)


solver = firedrake_ts.DAESolver(
    problem,
    solver_parameters=solver_params,
    monitor=monitor,
)

# Workaround for firedrake-ts callback issue with recent Firedrake/PETSc versions
solver._ctx._pre_jacobian_callback = None
solver._ctx._post_jacobian_callback = None
solver._ctx._pre_function_callback = None
solver._ctx._post_function_callback = None

solver.solve()

final_time = solver.ts.getTime()
write_output(final_time)

print("Done.")
print(f"Output written to result/q4_convection_Ra{Ra_value:.0e}.pvd")