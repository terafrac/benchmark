"""
This program solves Poisson's equation

    - div grad u(x, y) = -12

and boundary conditions given by

The exact solution is

u(x,y,z) = 1+x^2+2y^2+3z^2
"""

from dolfin import *

import sys


set_log_level(DEBUG)

# Create mesh
#n = 5
#mesh = UnitSquare(n, n)

# Create mesh 
box_x = 1.0
box_y = 2.0
box_z = 3.0
element_size = 0.5

num_elements_x = int(box_x / element_size)
num_elements_y = int(box_y / element_size)
num_elements_z = int(box_z / element_size)

mesh = BoxMesh(0, 0, 0, box_x, box_y, box_z, num_elements_x, num_elements_y, num_elements_z)
#print mesh
#plot(mesh, interactive=True)

# define function space
V = FunctionSpace(mesh, "Lagrange", 1)

# define boundary conditions
u0 = Expression('1+x[0]*x[0]+2*x[1]*x[1]+3*x[2]*x[2]')

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-12.0)
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc, solver_parameters={"linear_solver" : "direct"})

# Plot solution
#plot(u, interactive=True)

# exact solution interpolated onto mesh
u_exact = interpolate(u0, V)
u_exact_array = u_exact.vector().array()

# computed solution
u_array = u.vector().array()

# relative error
u_error = [0]*len(u_array)
for i in range(0, len(u_array)):
    u_error[i] = abs(u_array[i] / u_exact_array[i] - 1.0)

print "Maximum relative error is " + str(max(u_error))
if max(u_error) < 1e-10:
    print "OpenFrac benchmark passed."
    sys.exit(0)
else:
    print "OpenFrac Benchmark failed."
    sys.exit(1)

