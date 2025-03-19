import xarray as xr

# 打开本地文件
ds = xr.open_dataset("z500_2.5deg.nc")
ds2 = xr.open_dataset("F:/2023temperature and pressure/data_stream-oper_stepType-instant.nc")
print(ds)
print(ds2)

var_name = ['time']
print(ds2[var_name])

ds.close()
ds2.close()