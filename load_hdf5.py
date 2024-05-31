import dask.array as da
import dask.dataframe as dd
import h5py

file = h5py.File("test_atmos_cube.h5")
atmos = file["atmos"]
atmos_mapping = {}
for k in atmos.keys():
    atmos_mapping[k] = da.from_array(atmos[k], chunks="auto", name=k)