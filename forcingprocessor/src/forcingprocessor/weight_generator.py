import geopandas as gpd
import numpy as np
import xarray as xr
import rasterio
from rasterio.features import rasterize, geometry_window, bounds
import json, argparse
import datetime
import multiprocessing
from functools import partial

def process_row(row, src_string, y_extent):
    src = rasterio.open(src_string)
    window = geometry_window(src, [row['geometry']])
    row_offset, col_offset = window.row_off, window.col_off
    # Rasterize within the window
    geom_rasterize = rasterize(
        [(row["geometry"], 1)],
        out_shape=(window.height, window.width),
        transform=src.window_transform(window),
        all_touched=True,
        fill=0,
        dtype="uint8"
    )
    geom_rasterize = np.flipud(geom_rasterize)
    # Adjust local coordinates to global coordinates
    global_positions = np.where(geom_rasterize == 1)

    #print(global_positions)
    global_rows = global_positions[0] + (y_extent - row_offset)
    global_cols = global_positions[1] + col_offset

    global_coords = (global_rows, global_cols)
    return (row["divide_id"], global_coords)

def generate_weights_file(geopackage, grid_file, weights_filepath):
    try:
        src_string = f"netcdf:{grid_file}:RAINRATE"
        src = rasterio.open(src_string)
        ds = xr.open_dataset(grid_file,engine='netcdf4')
        grid = ds['RAINRATE']
    except Exception as e:
        raise Exception(f'\n\nError opening {grid_file}: {e}\n')

    g_df = gpd.read_file(geopackage, layer='divides')
    gdf_proj = g_df.to_crs(src.crs.to_string())
    # check transforms 
    print(src.transform)
    print(src.shape)
    print(grid.rio.transform())
    print(grid.rio.shape)
    # get the shape of the grid
    y_extent = grid.rio.shape[0]
    crosswalk_dict = {}
    start_time = datetime.datetime.now()
    print(f'Starting at {start_time}')
    rows = [row for _, row in gdf_proj.iterrows()]
    # Create a multiprocessing pool
    with multiprocessing.Pool() as pool:
        # Use a partial function to pass the constant 'grid' argument
        func = partial(process_row, src_string=src_string, y_extent=y_extent)
        # Map the function across all rows
        results = pool.map(func, rows)

    # Aggregate results
    for divide_id, global_coords in results:
        crosswalk_dict[divide_id] = global_coords


    weights_json = json.dumps(
        {k: [x.tolist() for x in v] for k, v in crosswalk_dict.items()}
    )
    print(f'Finished at {datetime.datetime.now()}')
    print(f'Total time: {datetime.datetime.now() - start_time}')
    with open(weights_filepath, "w") as f:
        f.write(weights_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("geopackage", type=str, help="Path to geopackage file")
    parser.add_argument("weights_filename", type=str, help="Filename for the weight file")
    parser.add_argument("grid_file", type=str, help="Path to grid file")
    args = parser.parse_args()   
    generate_weights_file(args.geopackage, args.grid_file, args.weights_filename)
