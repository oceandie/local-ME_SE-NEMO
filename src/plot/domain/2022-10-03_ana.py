"""Script to correctly plot Ana Aguiar's ORCA data, without land smearing."""

from cartopy.crs import PlateCarree
from iris import load
from iris.analysis.cartography import project
from iris.coords import AuxCoord
from iris import quickplot as qplt
from iris.util import squeeze
from matplotlib import colors
from matplotlib import pyplot as plt

my_cube_list = load("/data/users/aaguiar/immerse-monthly/orca12/2018_eke.grid_V_bounds.nc")
my_cube = my_cube_list.extract_cube("dmean_eke")
# Just one time step - can squeeze to just 2 dimensions.
my_cube = squeeze(my_cube)

coord_cubes = []
for coord_name in ("longitude", "latitude"):
    # Longitude and latitude are not correctly linked to dmean_eke, so need
    #  to manually convert them to coordinates.
    coord_cube = my_cube_list.extract_cube(coord_name)
    coord_kwargs = coord_cube.metadata._asdict()
    del coord_kwargs["cell_methods"]
    coord = AuxCoord(coord_cube.core_data(), **coord_kwargs)
    my_cube.add_aux_coord(coord, [0, 1])

# Project the data - seems to be necessary before plotting ORCA data.
#  https://scitools-iris.readthedocs.io/en/stable/generated/gallery/oceanography/plot_orca_projection.html
projected_cube, _ = project(my_cube, PlateCarree(), nx=1920, ny=1080)

qplt.pcolormesh(projected_cube, cmap='pink_r', norm=colors.LogNorm(vmin=1e-3, vmax=1e-1))
plt.gca().coastlines()
plt.savefig("tmp.png", dpi=300)
