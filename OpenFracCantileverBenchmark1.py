"""
Cantilever beam benchmark.

Beam has square cross section and uniform pressure load on top surface

All units are SI

Slender beam theory tip deflection is

w(tip) = 1.5 * (pressure / youngs_modulus) * width * (length / width)^4
"""

from dolfin import *

import sys

set_log_level(DEBUG)

# material properties
rho = 2200.0
youngs_modulus = 15e9
poisson_ratio = 0.1

mu = youngs_modulus / (2.0 * (1+poisson_ratio))
lmbda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio)*(1 - 2 * poisson_ratio))

beam_length = 100.0
beam_width = 2.0

top_pressure = 100.0

# Create mesh
box_x = beam_length
box_y = beam_width
box_z = beam_width
element_size = 0.5

num_elements_x = int(box_x / element_size)
num_elements_y = int(box_y / element_size)
num_elements_z = int(box_z / element_size)

mesh = BoxMesh(0, 0, -box_z, box_x, box_y, 0.0, num_elements_x, num_elements_y, num_elements_z)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

#plot(mesh, interactive=True)

# Dirichlet condition on right boundary
def boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-3;

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2] > -1e-3;

Gamma_Top = TopBoundary()
Gamma_Top.mark(boundary_parts, 0)

# pressure on top face
top_face_load = Constant((0,0,top_pressure))


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
    return 2*mu*epsilon(v) + lmbda*tr(epsilon(v))*Identity(v.cell().d)

# balance equation
F = inner(sigma(u), grad(v))*dx + inner(v,top_face_load)*ds(0)

# Extract bilinear and linear forms from F
a = lhs(F)
L = rhs(F)

A = assemble(a, exterior_facet_domains=boundary_parts)
b = assemble(L, exterior_facet_domains=boundary_parts)

# Set up PDE and solve
#info(LinearVariationalSolver.default_parameters(),1)

bc.apply(A, b)

u = Function(V)
U = u.vector()

solve(A, U, b, 'cg', 'ilu')

# Find average tip deflection
tip_deflection = 0.0
coordinates = mesh.coordinates()
#plot(u_z, interactive=True)
num_pts = 0
parameters["allow_extrapolation"] = True

for i in range(mesh.num_vertices()):
    if abs(coordinates[i][0] - beam_length) < DOLFIN_EPS:
        tip_deflection += u(coordinates[i])[2]
        num_pts += 1

tip_deflection /= num_pts

tip_deflection_model = -1.5*(top_pressure/youngs_modulus)*beam_width*pow(beam_length/beam_width, 4)
relative_error = abs(tip_deflection / tip_deflection_model - 1.0)

print "Relative tip deflection error is " + str(relative_error)
if relative_error < 5e-4:
    print "OpenFrac benchmark passed."
    sys.exit(0)
else:
    print "OpenFrac Benchmark failed."
    sys.exit(1)
