from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from scipy.interpolate import NearestNDInterpolator

from tqdm import tqdm
import pyvista as pv
import numpy as np

THRESHOLD = 150
BACKGROUND = 0
SPLINE_DEGREE = 4

volume = pv.ImageData('data/valid-1.nii')

mesh = volume.contour([THRESHOLD], method='flying_edges')
mesh.point_data.clear()

mesh.compute_normals(cell_normals=False, inplace=True)
mesh = mesh.extract_largest().clean().triangulate()

initial_point = (volume.bounds[1] / 2, 0, volume.bounds[-1] / 2 - 35)

p_ind = mesh.find_closest_point(initial_point)
p_normal = mesh.point_normals[p_ind]

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.add_points(mesh.points[[p_ind]], render_points_as_spheres=True, point_size=10)
plotter.show_axes()
plotter.show()

agreed_p_inx = np.where(np.dot(mesh.point_normals, p_normal) > 0)[0]
agreed_mesh = mesh.extract_points(agreed_p_inx).connectivity()

agreed_points_idx = agreed_mesh.point_data['vtkOriginalPointIds'].copy()
new_p_ind = np.where(agreed_points_idx == p_ind)[0]
p_region = agreed_mesh.point_data['RegionId'][new_p_ind]

region_point_ids = np.where(agreed_mesh.point_data['RegionId'] == p_region)[0]
region_mesh = agreed_mesh.extract_points(region_point_ids)

mesh.remove_points(agreed_points_idx[region_point_ids], inplace=True)

region_mesh = region_mesh.clean().extract_geometry()

random_points = region_mesh.points[np.random.randint(0, region_mesh.n_points, 1000)]
_, projections = mesh.find_closest_cell(random_points, return_closest_point=True)

region_mesh = region_mesh.translate((projections - random_points).mean(axis=0))

plotter = pv.Plotter()
plotter.add_mesh(mesh, color='red')
plotter.add_mesh(region_mesh, color='blue')
plotter.show()

# angle_mask = region_mesh.edge_mask(30)
# good_points_ids = np.where(~angle_mask)[0]
# good_mesh = region_mesh.extract_points(good_points_ids).extract_geometry()

X = region_mesh.points[:, [0, 2]]
Y = region_mesh.points[:, [1]]

model = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=SPLINE_DEGREE)),
    # ('spline', SplineTransformer(n_knots=SPLINE_DEGREE + 1, degree=SPLINE_DEGREE)),
    ('linear', RidgeCV(fit_intercept=True))
])

x, z = volume.bounds[1], volume.bounds[-1]
plane = pv.Plane(
    center=(x / 2, 0, z / 2), direction=(0, -1, 0),
    i_size=z, j_size=x, i_resolution=volume.dimensions[2] - 1, j_resolution=volume.dimensions[0] - 1
)

# model.fit(X, Y)
# Z = model.predict(plane.points[:, [0, 2]])

interp = NearestNDInterpolator(X, Y)
Z = interp(plane.points[:, [0, 2]])

plane.points[:, [1]] = Z
Z = Z.reshape(volume.dimensions[0], volume.dimensions[2]) / volume.spacing[1]
mask = np.ones(volume.dimensions[::-1])

plotter = pv.Plotter()
plotter.add_mesh(region_mesh)
plotter.add_mesh(plane)
plotter.show()

for i in tqdm(range(volume.dimensions[1])):
    J, I = np.where(Z + 1 >= i)
    mask[volume.dimensions[2] - I - 1, i, J] = BACKGROUND

volume['NIFTI'] = volume['NIFTI'] * mask.reshape(-1)
contour = volume.contour([THRESHOLD + 200]).extract_largest()

plotter = pv.Plotter()
plotter.add_mesh(contour)
# plotter.add_mesh(plane)
plotter.show()
