import xarray
from dask.distributed import Client


def test_fn(a):
    print(repr(a))
    return a

# out = xarray.apply_ufunc(
#     test_fn,
#     ds,
#     dask="parallelized",
# )
def run():
    out = xarray.map_blocks(
        test_fn,
        ds,
        template=ds,
    )
    out.compute()

if __name__ == "__main__":
    # NOTE(cmo): Chunks is needed to load into dask arrays
    ds = xarray.open_dataset("test_atmos_cube.nc", chunks={"x":1, "y": 1})

    client = Client(threads_per_worker=1)