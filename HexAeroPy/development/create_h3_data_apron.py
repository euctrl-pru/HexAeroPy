# Viz

import folium
from geojson import Feature, Point, FeatureCollection
import json
import pandas as pd
import matplotlib
import h3
import os.path

def hexagons_dataframe_to_geojson(df_hex, file_output=None):
    """
    Produce the GeoJSON for a dataframe, constructing the geometry from the "hex_id" column
    and including all other columns as properties.
    """
    list_features = []

    for i, row in df_hex.iterrows():
        try:
            geometry_for_row = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(h=row["hex_id"], geo_json=True)]}
            properties = row.to_dict()  # Convert all columns to a dictionary
            properties.pop("hex_id", None)  # Remove hex_id as it's already used in geometry
            feature = Feature(geometry=geometry_for_row, id=row["hex_id"], properties=properties)
            list_features.append(feature)
        except Exception as e:
            print(f"An exception occurred for hex {row['hex_id']}: {e}")

    feat_collection = FeatureCollection(list_features)
    geojson_result = json.dumps(feat_collection)
    return geojson_result


def get_color(custom_cm, val, vmin, vmax):
    return matplotlib.colors.to_hex(custom_cm((val-vmin)/(vmax-vmin)))

def choropleth_map(df_aggreg, column_name="value", border_color='black', fill_opacity=0.7, color_map_name="Blues", initial_map=None, initial_location=[47, 4], initial_zoom=5.5, tooltip_columns=None):
    """
    Creates choropleth maps given the aggregated data. 
    initial_map can be an existing map to draw on top of.
    initial_location and initial_zoom control the initial view of the map.
    tooltip_columns is a list of column names to display in a tooltip.
    """
    # colormap
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
    print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")

    # Create map if not provided
    if initial_map is None:
        initial_map = folium.Map(location=initial_location, zoom_start=initial_zoom, tiles="cartodbpositron")

    # Create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg)

    # Get colormap
    custom_cm = matplotlib.cm.get_cmap(color_map_name)

    # Add GeoJson to map
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': get_color(custom_cm, feature['properties'][column_name], vmin=min_value, vmax=max_value),
            'color': border_color,
            'weight': 1,
            'fillOpacity': fill_opacity
        },
        tooltip=folium.features.GeoJsonTooltip(fields=tooltip_columns) if tooltip_columns else None,
        name="Choropleth"
    ).add_to(initial_map)

    return initial_map







# Dev

import pandas as pd
import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import transform
from functools import partial
from pyproj import Transformer
import h3
import re
import glob

airports_df = pd.read_csv('https://davidmegginson.github.io/ourairports-data/airports.csv')

# Filter the DataFrame
airports_df = airports_df[
    (airports_df['type'].isin(['large_airport', 'medium_airport']))][['ident', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'type']]

print(f"There are {len(airports_df)} airports to process...")

def line_to_polygon(line, width_m):
    """
    Convert a LineString to a Polygon with a specified width in meters.

    Parameters:
    line (LineString): The LineString object to be converted.
    width_m (float): The width of the line in meters.

    Returns:
    Polygon: A Polygon object representing the LineString with the specified width.
    """
    # Define the projection transformer
    # WGS 84 (latitude and longitude) to World Mercator projection
    transformer_to_meters = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    transformer_to_degrees = Transformer.from_crs("EPSG:3395", "EPSG:4326", always_xy=True)

    # Transform the LineString to World Mercator projection (meters)
    line_in_meters = transform(transformer_to_meters.transform, line)

    # Buffer the line in meters
    buffered_line = line_in_meters.buffer(width_m)

    # Transform the buffered line back to WGS 84 (degrees)
    buffered_line_in_degrees = transform(transformer_to_degrees.transform, buffered_line)

    return buffered_line_in_degrees

def lines_to_polygons(lines, standard_width):
    """
    Convert a collection of LineStrings to Polygons with a specified width in meters.

    Parameters:
    lines [((LineString), (float))]: The LineString object to be converted and 
    the width of the line in meters.
    standard_width float: When the width is not available, this standard width will be used.
    
    Returns:
    array<Polygon>: An array of Polygon objects representing the LineString with the specified width.
    """
    polygons = []
    for x in lines:
        line = x[0]
        width = x[1]
        if pd.isna(width):
            width = standard_width
        width = width
        polygons.append(line_to_polygon(line, width))
    return polygons


def query_features(lat, lon, feature_type, radius=5000):
    """
    Query OpenStreetMap for aeroway features and return a DataFrame with metadata and geometries.

    Args:
    lat (float): Latitude of the center point.
    lon (float): Longitude of the center point.
    feature_type (str): Type of the aeroway feature (e.g., 'runway', 'taxiway').
    radius (int): Radius in meters around the point to search for the feature.

    Returns:
    DataFrame: A DataFrame with columns for metadata and geometry.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    way["aeroway"="{feature_type}"](around:{radius},{lat},{lon});
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    features = []
    for element in data['elements']:
        if element['type'] == 'way':
            nodes = {node['id']: (node['lat'],node['lon']) for node in data['elements'] if node['type'] == 'node'}
            points = [nodes[node_id] for node_id in element['nodes'] if node_id in nodes]
            linestring = LineString(points)

            feature_data = element.get('tags', {})
            feature_data.update({
                'id': element.get('id'),
                'type': element.get('type'),
                'geometry': linestring
                # Add other metadata fields here if needed
            })
            features.append(feature_data)

    return pd.DataFrame(features)

def shapely_polygon_to_geojson(polygon):
    """
    Convert a Shapely Polygon to a GeoJSON-like dictionary.

    Parameters:
    polygon (shapely.geometry.polygon.Polygon): The Shapely Polygon to convert.

    Returns:
    dict: A dictionary representing the Polygon in GeoJSON format.
    """
    # Extract the exterior of the polygon and convert it to a list of coordinates
    exterior_coords = list(polygon.exterior.coords)
    
    # Create a GeoJSON-like structure
    geojson_polygon = {
        "type": "Polygon",
        "coordinates": [exterior_coords]
    }

    return geojson_polygon

def polygon_to_h3(poly, resolution):
    """
    Convert a Shapely Polygon to a set of H3 indices.

    Parameters:
    poly (shapely.geometry.polygon.Polygon): The Shapely Polygon to convert.
    resolution (int): The H3 resolution.

    Returns:
    set: A set of H3 indices covering the Polygon.
    """
    geojson_poly = shapely_polygon_to_geojson(poly)
    return h3.polyfill(geojson_poly, resolution)

def is_number(s):
    """
    Checks if the input string is a valid number.
    
    Parameters:
    - s: The string to check.
    
    Returns:
    - True if the string is a number, False otherwise.
    """
    try:
        if pd.isna(s):
            return None
            
        float(s)  # Try converting the string to float
        return True
    except ValueError:
        return False

def safe_convert_to_float(s):
    """
    Safely converts a string to a float by first checking if it's already a valid number.
    If not, it removes non-numeric characters (except for the decimal point) and then tries to convert.
    
    Parameters:
    - s: The string to convert.
    
    Returns:
    - A float representation of the string if conversion is possible, None otherwise.
    """
    if is_number(s):
        return float(s)  # Return the float value directly if it's already a number
    
    if pd.isna(s):
        return None
    
    # If not a valid number, try removing non-numeric characters and convert
    
    numeric_part = re.sub(r"[^-0-9.]+", "", s)
    
    try:
        if pd.isna(numeric_part):
            return None
        return float(numeric_part)
    except ValueError:
        return None

def hexagonify_airport(
    apt_icao, 
    radius,
    airports_df,
    resolution = 12,
    standard_width_runways = 50, # In case OSM does not have the width of the object this is the standard value
    standard_width_taxiways = 20,
    standard_width_parking = 20,
    mp_width_runways = 1, # In case you need a buffer around your object, multiply > 1. 
    mp_width_taxiways = 1,
    mp_width_parking = 1):
    
    latitude = airports_df[airports_df['ident'] == apt_icao].latitude_deg.values[0]
    longitude = airports_df[airports_df['ident'] == apt_icao].longitude_deg.values[0]

    runways_df = query_features(latitude, longitude, 'runway', radius)
    taxiways_df = query_features(latitude, longitude, 'taxiway', radius)
    parking_positions_df = query_features(latitude, longitude, 'parking_position', radius)
    
    #print(f"runways_df: len = {len(runways_df)}, columns = {list(runways_df.columns)}")
    #print(f"taxiways_df: len = {len(taxiways_df)}, columns = {list(taxiways_df.columns)}")
    #print(f"parking_positions_df: len = {len(parking_positions_df)}, columns = {list(parking_positions_df.columns)}")
    #print()
    
    if len(runways_df) == 0:
        runways_df['geometry'] = None
    if len(taxiways_df) == 0:
        taxiways_df['geometry'] = None
        
    if len(parking_positions_df) == 0:
        parking_positions_df['geometry'] = None
    
    if 'width' not in runways_df.columns:
        runways_df['width'] = None

    if 'width' not in taxiways_df.columns:
        taxiways_df['width'] = None

    if 'width' not in parking_positions_df.columns:
        parking_positions_df['width'] = None
    
    runways_df['width'] = runways_df['width'].apply(safe_convert_to_float)
    runways_df['width'] = runways_df['width'].apply(lambda l: None if pd.isna(l) else l * mp_width_runways)
    
    taxiways_df['width'] = taxiways_df['width'].apply(safe_convert_to_float)
    taxiways_df['width'] = taxiways_df['width'].apply(lambda l: None if pd.isna(l) else l * mp_width_runways)
    
    parking_positions_df['width'] = parking_positions_df['width'].apply(safe_convert_to_float)
    parking_positions_df['width'] = parking_positions_df['width'].apply(lambda l: None if pd.isna(l) else l * mp_width_runways)
    
    runways_df['polygon'] = lines_to_polygons(list(zip(runways_df.geometry.to_list(), runways_df.width.to_list())), standard_width_runways*mp_width_runways)
    taxiways_df['polygon'] = lines_to_polygons(list(zip(taxiways_df.geometry.to_list(), taxiways_df.width.to_list())), standard_width_taxiways*mp_width_taxiways)
    parking_positions_df['polygon'] = lines_to_polygons(list(zip(parking_positions_df.geometry.to_list(), parking_positions_df.width.to_list())), standard_width_parking*mp_width_parking)

    runways_df['h3_res10'] = runways_df['polygon'].apply(lambda l: polygon_to_h3(l, resolution))
    taxiways_df['h3_res10'] = taxiways_df['polygon'].apply(lambda l: polygon_to_h3(l, resolution))
    parking_positions_df['h3_res10'] = parking_positions_df['polygon'].apply(lambda l: polygon_to_h3(l, resolution))

    runways_df['color_type'] = 1 
    taxiways_df['color_type'] = 5000 
    parking_positions_df['color_type'] = 10000 

    df = pd.concat([runways_df, taxiways_df, parking_positions_df])
    df = df.explode('h3_res10').rename({'h3_res10':'hex_id'},axis=1)
    df = df[~df.hex_id.isna()]

    df['apt_icao'] = apt_icao
    df['hex_latitude'], df['hex_longitude'] = zip(*df['hex_id'].apply(h3.h3_to_geo)) 
    df['hex_res'] = resolution
    return df, latitude, longitude


for apt_icao in airports_df.ident.to_list():
    print(f"Processing {apt_icao}...")
    # Radius around which airport elements are searched within OSM
    radius = 5000 
    res = 12
    if os.path.isfile(f'data/apron_hex/h3_res_{res}_apron_{apt_icao}.parquet'):
      print('FILE EXISTS - Skipping...')
      print()
      continue

    try: 
        df, latitude, longitude = hexagonify_airport(
            apt_icao, 
            radius,
            airports_df,
            resolution = res,
            standard_width_runways = 50, # In case OSM does not have the width of the object this is the standard value
            standard_width_taxiways = 30,
            standard_width_parking = 25,
            mp_width_runways = 1.5, # In case you need a buffer around your object, multiply > 1. 
            mp_width_taxiways = 1.5,
            mp_width_parking = 1.5)
        
        s = ['apt_icao', 'hex_id', 'hex_latitude', 'hex_longitude', 'hex_res']
        df[s + [x for x in df.columns if x not in s + ['geometry', 'polygon']]].to_parquet(f'data/apron_hex/h3_res_{res}_apron_{apt_icao}.parquet')
        print()
    except Exception as e: 
        print(f"Failed to process {apt_icao}. Error: {e}")
        print()
        continue
    
    try:
      print(f"Creating viz for {apt_icao}...")
      
      columns = ['aeroway', 'width', 'id', 'color_type','hex_id'] 
      tt_columns = ['aeroway', 'width', 'id']
      
      if 'length' in df.columns:
        columns = columns + ['length']
        tt_columns = tt_columns + ['length']
        
      if 'ref' in df.columns:
        columns = columns + ['ref']
        tt_columns = tt_columns + ['ref']
      
      if 'surface' in df.columns: 
        columns = columns + ['surface']
        tt_columns = tt_columns + ['surface']
      
      if len(df) < 10000:
        df_viz = df[columns]

        m = choropleth_map(
            df_viz,
            column_name='color_type',
            border_color='black',
            fill_opacity=0.7,
            color_map_name='Reds',
            initial_map=None,
            initial_location=[latitude, longitude],
            initial_zoom = 14,
            tooltip_columns = tt_columns
        )

        m.save(f'data/apron_hex_viz/h3_res_{res}_apron_{apt_icao}_visualization.html')
    except Exception as e:
      print(f"Visualization failed for {apt_icao}. Error: {e}")
      print()
    
# Construct the file pattern to match
file_pattern = f"data/apron_hex/*"

# Use glob to find all files matching the pattern
files = glob.glob(file_pattern)

df_list = [pd.read_parquet(file) for file in files]
concatenated_df = pd.concat(df_list, ignore_index=True)

# Output the concatenated DataFrame to a single Parquet file
output_file = f"data/apron_hex/h3_res_{res}_apron_all_airports.parquet"
concatenated_df.to_parquet(output_file)