# Solve the biharmonic system with RHS f = dT/dx
# where T(x,y) = (1-y) + A cos(pi x), A = 0.1

from firedrake import *
import os

# Mesh
N = 128
mesh = UnitSquareMesh(N, N, quadrilateral=True)

# Function spaces
V = FunctionSpace(mesh, "Lagrange", 1)
ME = MixedFunctionSpace([V, V], name=["vorticity", "streamfunction"])
Vv = VectorFunctionSpace(mesh, "Lagrange", 1)

# Test functions
omega_t, psi_t = TestFunctions(ME)

# Unknowns
u = Function(ME)
u.subfunctions[0].rename("vorticity")
u.subfunctions[1].rename("streamfunction")

omega, psi = split(u)

# Coordinates
x, y = SpatialCoordinate(mesh)

# Temperature field
A = Constant(0.1)
T = Function(V, name="temperature")
T.interpolate((1.0 - y) + A*cos(pi*x))

# RHS f = dT/dx
f = T.dx(0)

# Weak form:
# -Delta omega = dT/dx
# -Delta psi = omega
Fomega = inner(grad(omega_t), grad(omega))*dx - omega_t*f*dx
Fpsi = inner(grad(psi_t), grad(psi))*dx - psi_t*omega*dx
F = Fomega + Fpsi

# Boundary conditions: omega = 0, psi = 0 on boundary
bcs = [
    DirichletBC(ME.sub(0), 0.0, "on_boundary"),
    DirichletBC(ME.sub(1), 0.0, "on_boundary")
]

# Solver parameters
params = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij",
}

# Solve
problem = NonlinearVariationalProblem(F, u, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.solve()

# Compute velocity v = curl(psi k)
psi_sol = u.subfunctions[1]
v = Function(Vv, name="velocity")
v.interpolate(curl(psi_sol))

# Output for ParaView
os.makedirs("result", exist_ok=True)
outfile = VTKFile("result/q3_biharm_temperature.pvd")
outfile.write(T, u.subfunctions[0], u.subfunctions[1], v)

print("Done.")
print("Output written to result/q3_biharm_temperature.pvd")