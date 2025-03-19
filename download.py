import xarray as xr
import gcsfs
import numpy as np
from tqdm import tqdm
import dask
import gc
from datetime import datetime
import os
# 配置Dask禁用分块警告
dask.config.set(**{'array.slicing.split_large_chunks': False})

# 需要特殊处理的变量配置
VAR_CONFIG = {
    'level_vars': [  # 需要选择500hPa的变量
        'geopotential', 'specific_humidity', 'potential_vorticity',
        'temperature', 'u_component_of_wind', 'v_component_of_wind'
    ],
    'static_vars':[ #时间无关的变量
        'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography',
        'geopotential_at_surface', 'high_vegetation_cover', 'lake_cover', 'land_sea_mask',
        'low_vegetation_cover', 'slope_of_sub_gridscale_orography','soil_type',
        'type_of_high_vegetation', 'type_of_low_vegetation'
    ]
}
dynamic_vars = ['10m_u_component_of_wind',
               '10m_v_component_of_wind',
               '2m_dewpoint_temperature',
               '2m_temperature',
               'boundary_layer_height',
               'geopotential',
               'leaf_area_index_high_vegetation',
               'leaf_area_index_low_vegetation',
               'mean_sea_level_pressure', 'mean_surface_latent_heat_flux',
               'mean_surface_net_long_wave_radiation_flux',
               'mean_surface_net_short_wave_radiation_flux',
               'mean_surface_sensible_heat_flux', 'mean_top_downward_short_wave_radiation_flux',
               'mean_top_net_long_wave_radiation_flux', 'mean_top_net_short_wave_radiation_flux',
               'mean_vertically_integrated_moisture_divergence', 'potential_vorticity',
               'sea_ice_cover', 'sea_surface_temperature',
                'snow_depth',
               'specific_humidity', 'standard_deviation_of_filtered_subgrid_orography',
               'standard_deviation_of_orography', 'surface_pressure', 'temperature',
               'total_cloud_cover', 'total_column_water', 'total_column_water_vapour',
               'total_precipitation',
               'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
               'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
               'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4']


def process_static_vars(zarr_path, output_path):
    """预处理并保存静态变量"""
    fs = gcsfs.GCSFileSystem(anon=True)
    store = fs.get_mapper(zarr_path)

    with xr.open_zarr(store, chunks={}) as ds:
        # 创建包含完整维度定义的模板
        template = xr.Dataset(
            coords={
                "latitude": np.arange(90, -90.1, -2.5),
                "longitude": np.arange(0, 360, 2.5),
                "time": []
            }
        )
        # 初始化文件时定义时间维度为unlimited ↓
        template.to_netcdf(
            output_path,
            mode="w",
            format="NETCDF4",
            unlimited_dims=["time"],
            encoding = {"time": {"dtype": "datetime64[ns]"}}
        )

        # 写入静态变量到根组
        static_ds = xr.Dataset()
        for var in VAR_CONFIG['static_vars']:
            var_data = ds[var]
            if 'latitude' in var_data.dims and 'longitude' in var_data.dims:
                var_data = var_data.isel(
                    latitude=slice(None, None, 10),
                    longitude=slice(None, None, 10)
                )
            static_ds[var] = var_data

        static_ds.to_netcdf(
            output_path,
            mode="a",
            group=None,  # 写入根组
            encoding={var: {"zlib": True} for var in static_ds.data_vars}
        )


def process_dynamic_chunk(time_chunk, fs, zarr_path, output_path):
    """处理动态变量时间分块"""
    store = fs.get_mapper(zarr_path)
    ds = None
    ds_chunk = None
    processed_ds = None
    try:
        # 1. 优化分块读取
        ds = xr.open_zarr(
            store,
            chunks={"time": 24}
        )


        print(f"开始处理时间段: {time_chunk.start} 至 {time_chunk.stop}")


        # 2. 筛选动态变量
        ds_chunk = ds[dynamic_vars].sel(time=time_chunk)
        ds_chunk = ds_chunk.sel(time=ds_chunk.time.dt.hour == 0)

        # 3. 处理各变量
        processed_vars = {}
        for var in ds_chunk.data_vars:
            # 层次选择
            if var in VAR_CONFIG['level_vars']:
                var_data = ds_chunk[var].sel(level=500, method="nearest")
            else:
                var_data = ds_chunk[var]

            # 空间降采样
            var_data = var_data.isel(
                latitude=slice(None, None, 10),
                longitude=slice(None, None, 10)
            )

            processed_vars[var] = var_data


            # 4创建处理后的Dataset
        processed_ds = xr.Dataset(processed_vars).chunk({
                "time": 24,
                "latitude": 73,
                "longitude": 144
            })

        # 5. 设置编码
        encoding = {
            var: {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": 9.96921e+36
            } for var in processed_ds.data_vars
        }

        # 6. 写入文件
        processed_ds.compute().to_netcdf(
            output_path,
            mode="a",
            group="dynamic",
            encoding=encoding,
            unlimited_dims=["time"]
        )

    finally:
        if ds is not None:
            del ds
        if ds_chunk is not None:
            del ds_chunk
        if processed_ds is not None:
            del processed_ds
        gc.collect()


def main():
    zarr_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
    output_path = "era5_full_vars_optimized.nc"
    fs = gcsfs.GCSFileSystem(anon=True)
    # 第一步：处理静态变量
    if not os.path.exists(output_path):
        process_static_vars(zarr_path, output_path)

        # 按年处理动态变量
    years = range(1959, 2024)
    with tqdm(years, desc="Processing") as pbar:
        for year in pbar:
            time_slice = slice(f"{year}-01-01", f"{year}-12-31")
            try:
                process_dynamic_chunk(time_slice, fs, zarr_path, output_path)
                pbar.set_postfix_str(f"{year} 完成")
            except Exception as e:
                print(f"{year} 年处理失败: {str(e)}")
                continue

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    print(f"总处理时间: {datetime.now() - start_time}")