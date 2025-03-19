import gcsfs
import xarray as xr

fs = gcsfs.GCSFileSystem(token="anon")
store = fs.get_mapper("gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")
ds = xr.open_zarr(store)

# 查看原始分块信息
print(ds.chunks)