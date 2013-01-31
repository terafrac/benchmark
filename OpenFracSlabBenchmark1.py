"""
Slab elasticity benchmark.

Slab is 1000X1000 meters and 100 meters thick and is
being compressed under its own weight

Bottom surface is fixed

All units are SI

Displacement is in Z direction only

w(z) = rho*g/(2*(2G+lambda))*z*(z-2*h)

h is thickness

Mesh is second order tetrahedral elements
"""

from dolfin import *

import sys

set_log_level(DEBUG)

# material properties
youngs_modulus = 15e9
poisson_ratio = 0.1
rho = 2200.0
g = 9.8

shear_modulus = youngs_modulus / (2.0 * (1+poisson_ratio))
lmbda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio)*(1 - 2 * poisson_ratio))

slab_width = 1000.0
slab_thickness = 100.0


# Create mesh
box_x = slab_width
box_y = slab_width
box_z = slab_thickness
element_size = 25.0

num_elements_x = int(box_x / element_size)
num_elements_y = int(box_y / element_size)
num_elements_z = int(box_z / element_size)

mesh = BoxMesh(-0.5*box_x, -0.5*box_y, 0.0, 0.5*box_x, 0.5*box_y, box_z, num_elements_x, num_elements_y, num_elements_z)

#plot(mesh, interactive=True)

# Dirichlet condition on bottom boundary
def boundary(x, on_boundary):
    return on_boundary and x[2] < 1e-3;

# pressure on top face
gravity = Constant((0,0,-rho*g))

# Define function space
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

c = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, c, boundary)

# Strain
def epsilon(v):
    return sym(grad(v))

# Stress
def sigma(v):
    return 2*shear_modulus*epsilon(v) + lmbda*tr(epsilon(v))*Identity(v.cell().d)

# balance equation
F = inner(sigma(u), grad(v))*dx - inner(v,gravity)*dx

# Extract bilinear and linear forms from F
a = lhs(F)
L = rhs(F)

A = assemble(a)
b = assemble(L)

# Set up PDE and solve
#info(LinearVariationalSolver.default_parameters(),1)

bc.apply(A, b)

u = Function(V)
U = u.vector()

solve(A, U, b, 'cg', 'ilu')

# Find average tip deflection
top_deflection = 0.0
coordinates = mesh.coordinates()
#plot(u_z, interactive=True)
num_pts = 0
parameters["allow_extrapolation"] = True

for i in range(mesh.num_vertices()):
    if abs(coordinates[i][2] - slab_thickness) < DOLFIN_EPS:
        top_deflection += u(coordinates[i])[2]
        num_pts += 1

top_deflection /= num_pts

top_deflection_model = -rho*g*pow(slab_thickness,2)/(2*(2*shear_modulus+lmbda))
relative_error = abs(top_deflection / top_deflection_model - 1.0)

print "Relative top deflection error is " + str(relative_error)
print "Top deflection is " + str(top_deflection)
print "Model deflection is" + str(top_deflection_model)

if relative_error < 5e-3:
    print "OpenFrac benchmark passed."
    sys.exit(0)
else:
    print "OpenFrac Benchmark failed."
    sys.exit(1)
