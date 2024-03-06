import folium
from geojson import Feature, Point, FeatureCollection
import json
import pandas as pd
import matplotlib
import h3

pd.options.mode.chained_assignment = None


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















from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,from_unixtime, min, max, to_date, pandas_udf, col, PandasUDFType, lit, round
from pyspark.sql.types import DoubleType, StructType, StructField
from pyspark.sql import functions as F
from pyspark.sql import Window

import os, time
import subprocess
import os,shutil
from datetime import datetime
import pandas as pd
import numpy as np
from IPython.display import display, HTML

import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import transform
import pyproj
from functools import partial
from shapely.geometry import LineString
from shapely.ops import transform
from pyproj import Proj, Transformer
import pandas as pd
import folium
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely.geometry
import h3
from sklearn.preprocessing import LabelEncoder

# Settings
project = "project_aiu"

# Getting today's date
today = datetime.today().strftime('%d %B %Y')

# Spark Session Initialization
shutil.copy("/runtime-addons/cmladdon-2.0.40-b150/log4j.properties", "/etc/spark/conf/") # Setting logging properties

spark = SparkSession.builder \
    .appName("RWY Identification Grid Maker") \
    .config("spark.log.level", "ERROR")\
    .config("spark.hadoop.fs.azure.ext.cab.required.group", "eur-app-aiu-dev") \
    .config("spark.kerberos.access.hadoopFileSystems", "abfs://storage-fs@cdpdldev0.dfs.core.windows.net/data/project/aiu.db/unmanaged") \
    .config("spark.driver.cores", "1") \
    .config("spark.driver.memory", "8G") \
    .config("spark.executor.memory", "5G") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.instances", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "6") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "400s") \
    .enableHiveSupport() \
    .getOrCreate()

# Get environment variables
engine_id = os.getenv('CDSW_ENGINE_ID')
domain = os.getenv('CDSW_DOMAIN')

# Format the URL
url = f"https://spark-{engine_id}.{domain}"

# Display the clickable URL
display(HTML(f'<a href="{url}">{url}</a>'))

airports_df = spark.sql(f"""
    SELECT ident, latitude_deg, longitude_deg, elevation_ft, type
    FROM {project}.oa_airports
    WHERE type = 'large_airport' OR type = 'medium_airport';
""").toPandas()

runways_df = spark.sql(f"""
    SELECT *
    FROM {project}.oa_runways
""")


from pyspark.sql.functions import udf, explode
from pyspark.sql.types import ArrayType, StringType
import h3
import math
import json

def calculate_new_lat_lon(lat, lon, distance, bearing):
  
    if pd.isna(lat) or pd.isna(lon) or pd.isna(distance) or pd.isna(bearing):
        return None, None
  
    R = 6378.1  # Radius of the Earth in km
    bearing = math.radians(bearing)  # Convert bearing to radians
    lat = math.radians(lat)  # Convert latitude to radians
    lon = math.radians(lon)  # Convert longitude to radians

    new_lat = math.asin(math.sin(lat) * math.cos(distance / R) +
                        math.cos(lat) * math.sin(distance / R) * math.cos(bearing))
    new_lon = lon + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat),
                               math.cos(distance / R) - math.sin(lat) * math.sin(new_lat))

    new_lat = math.degrees(new_lat)
    new_lon = math.degrees(new_lon)
    return new_lat, new_lon

def create_runway_polygon(lat, lon, length, width, heading, runway_width_mp = 2):
    try:
      length_km = length * 0.0003048  # Convert length from feet to km
      width_km = width * 0.0003048  # Convert width from feet to km

      opposite_end_lat, opposite_end_lon = calculate_new_lat_lon(lat, lon, length_km, heading)
      corner1_lat, corner1_lon = calculate_new_lat_lon(lat, lon, width_km / 2 * runway_width_mp, heading + 90)
      corner2_lat, corner2_lon = calculate_new_lat_lon(lat, lon, width_km / 2 * runway_width_mp, heading - 90)
      corner3_lat, corner3_lon = calculate_new_lat_lon(opposite_end_lat, opposite_end_lon, width_km / 2 * runway_width_mp, heading + 90)
      corner4_lat, corner4_lon = calculate_new_lat_lon(opposite_end_lat, opposite_end_lon, width_km / 2 * runway_width_mp, heading - 90)

      polygon = {
          "type": "Polygon",
          "coordinates": [[
              [corner1_lon, corner1_lat], [corner2_lon, corner2_lat],
              [corner4_lon, corner4_lat], [corner3_lon, corner3_lat],
              [corner1_lon, corner1_lat]
          ]]
      }
      return json.dumps(polygon)
    
    except Exception as e: 
      print(f'Error in create_runway_polygon: {e}')
      return None

def create_single_approach_polygon(lat, lon, width_km, approach_length_km, max_approach_length_km, far_end_width_km, heading, runway_width_mp = 2):
    try:
      approach_length_km_close = approach_length_km - 1.852 ## Minus 1 NM
      approach_length_km_far = approach_length_km

      near_end_lat, near_end_lon = calculate_new_lat_lon(lat, lon, approach_length_km_close, heading)
      far_end_lat, far_end_lon = calculate_new_lat_lon(lat, lon, approach_length_km_far, heading)

      near_corner1_lat, near_corner1_lon = calculate_new_lat_lon(lat, lon, width_km / 2 * runway_width_mp, heading + 90)
      near_corner2_lat, near_corner2_lon = calculate_new_lat_lon(lat, lon, width_km / 2 * runway_width_mp, heading - 90)

      near_corner1_lat, near_corner1_lon = calculate_new_lat_lon(
          near_end_lat, 
          near_end_lon, 
          (far_end_width_km * (approach_length_km_close/approach_length_km_far) + width_km) / 2 * runway_width_mp, 
          heading + 90)

      near_corner2_lat, near_corner2_lon = calculate_new_lat_lon(
          near_end_lat, 
          near_end_lon, 
          (far_end_width_km * (approach_length_km_close/approach_length_km_far) + width_km) / 2 * runway_width_mp, 
          heading - 90)

      far_corner1_lat, far_corner1_lon = calculate_new_lat_lon(far_end_lat, far_end_lon, far_end_width_km / 2 * runway_width_mp, heading + 90)
      far_corner2_lat, far_corner2_lon = calculate_new_lat_lon(far_end_lat, far_end_lon, far_end_width_km / 2 * runway_width_mp, heading - 90)

      approach_polygon = {
          "type": "Polygon",
          "coordinates": [[
              [near_corner1_lon, near_corner1_lat], [near_corner2_lon, near_corner2_lat],
              [far_corner2_lon, far_corner2_lat], [far_corner1_lon, far_corner1_lat],
              [near_corner1_lon, near_corner1_lat]
          ]]
      }
      return json.dumps(approach_polygon)
    except Exception as e: 
      print(f'Error in create_single_approach_polygon: {e}')
      return None
    
def create_low_numbered_approach_area_polygons(lat_le, lon_le, lat_he, lon_he, length, width, heading, approach_length_nmi=10, max_approach_length_nmi = 10, far_end_width_factor=25):
    length_km = length * 0.0003048  # Convert length from feet to km
    width_km = width * 0.0003048  # Convert width from feet to km
    approach_length_km = approach_length_nmi * 1.852
    far_end_width_km = width_km * far_end_width_factor

    # Create trapezoid for the lower-numbered end
    le_end_lat, le_end_lon = calculate_new_lat_lon(lat_le, lon_le, length_km, heading)
    polygons_low_numbered = create_single_approach_polygon(le_end_lat, le_end_lon, width_km, approach_length_km, max_approach_length_nmi, far_end_width_km, heading)
    return polygons_low_numbered

def create_high_numbered_approach_area_polygons(lat_le, lon_le, lat_he, lon_he, length, width, heading, approach_length_nmi, max_approach_length_nmi = 10, far_end_width_factor=25):
    length_km = length * 0.0003048  # Convert length from feet to km
    width_km = width * 0.0003048  # Convert width from feet to km
    approach_length_km = approach_length_nmi * 1.852
    far_end_width_km = width_km * far_end_width_factor

    # Create trapezoid for the higher-numbered end
    opposite_heading = (heading + 180) % 360
    he_end_lat, he_end_lon = calculate_new_lat_lon(lat_he, lon_he, length_km, opposite_heading)
    polygons_high_numbered = create_single_approach_polygon(he_end_lat, he_end_lon, width_km, approach_length_km, max_approach_length_nmi, far_end_width_km, opposite_heading)

    return polygons_high_numbered

def fill_polygon_with_hexagons(polygon_json, resolution=11):
    polygon = json.loads(polygon_json)
    hexagons = h3.polyfill(polygon, resolution, geo_json_conformant=True)
    return list(hexagons)

create_runway_polygon_udf = udf(create_runway_polygon, StringType())
create_low_numbered_approach_area_polygons_udf = udf(create_low_numbered_approach_area_polygons, ArrayType(StringType()))
create_high_numbered_approach_area_polygons_udf = udf(create_high_numbered_approach_area_polygons, ArrayType(StringType()))
fill_hexagons_udf = udf(fill_polygon_with_hexagons, ArrayType(StringType()))

def create_hex_airport(icao_apt):
    apt_runways_df = runways_df.filter(runways_df.airport_ident == icao_apt)
    
    # Example average values for runway length and width
    default_length_ft = 7000  # Default average length in feet
    default_width_ft = 150  # Default average width in feet
    
    # Fill null values for length and width with default average values
    apt_runways_df = apt_runways_df.na.fill({
        'length_ft': default_length_ft,
        'width_ft': default_width_ft
    })
    
    apt_runways_df = apt_runways_df.withColumn(
        "runway_polygon",
        create_runway_polygon_udf(
            apt_runways_df.le_latitude_deg,
            apt_runways_df.le_longitude_deg,
            apt_runways_df.length_ft,
            apt_runways_df.width_ft,
            apt_runways_df.le_heading_degT
        )
    ).filter(col("runway_polygon").isNotNull())

    apt_runways_df = apt_runways_df.withColumn(
        "runway_hexagons",
        fill_hexagons_udf("runway_polygon")
    )


    max_approach_length_nmi=10
    far_end_width_factor = 25

    for distance_nm in range(1,max_approach_length_nmi+1):
        apt_runways_df = apt_runways_df.withColumn(
            f"low_numbered_approach_polygons_distance_{distance_nm}_nm",
            create_low_numbered_approach_area_polygons_udf(
                apt_runways_df.le_latitude_deg,
                apt_runways_df.le_longitude_deg,
                apt_runways_df.he_latitude_deg,
                apt_runways_df.he_longitude_deg,
                apt_runways_df.length_ft,
                apt_runways_df.width_ft,
                apt_runways_df.le_heading_degT,
                lit(distance_nm), 
                lit(max_approach_length_nmi),
                lit(distance_nm/max_approach_length_nmi*far_end_width_factor)    
            )
        ).withColumn(
            f"high_numbered_approach_polygons_distance_{distance_nm}_nm",
            create_high_numbered_approach_area_polygons_udf(
                apt_runways_df.le_latitude_deg,
                apt_runways_df.le_longitude_deg,
                apt_runways_df.he_latitude_deg,
                apt_runways_df.he_longitude_deg,
                apt_runways_df.length_ft,
                apt_runways_df.width_ft,
                apt_runways_df.le_heading_degT,
                lit(distance_nm), 
                lit(max_approach_length_nmi),
                lit(distance_nm/max_approach_length_nmi*far_end_width_factor)
            )
        ).filter(
          col(f"low_numbered_approach_polygons_distance_{distance_nm}_nm",).isNotNull()).filter(
          col(f"high_numbered_approach_polygons_distance_{distance_nm}_nm",).isNotNull()).withColumn(
            f"low_numbered_approach_hexagons_{distance_nm}_nm",
            fill_hexagons_udf(f"low_numbered_approach_polygons_distance_{distance_nm}_nm")
        ).withColumn(
            f"high_numbered_approach_hexagons_{distance_nm}_nm",
            fill_hexagons_udf(f"high_numbered_approach_polygons_distance_{distance_nm}_nm")
        )

    df = apt_runways_df.toPandas()

    # Assuming `df` is your DataFrame
    # Specify your ID columns and columns to melt
    id_vars = ['id', 'airport_ref', 'airport_ident', 'length_ft', 'width_ft',
               'surface', 'lighted', 'closed', 'le_ident', 'le_latitude_deg',
               'le_longitude_deg', 'le_elevation_ft', 'le_heading_degT',
               'le_displaced_threshold_ft', 'he_ident', 'he_latitude_deg',
               'he_longitude_deg', 'he_elevation_ft', 'he_heading_degT',
               'he_displaced_threshold_ft']

    # Columns to melt (based on your list)
    value_vars = [
        'runway_hexagons',
        'runway_hexagons',
        'low_numbered_approach_hexagons_1_nm',
        'high_numbered_approach_hexagons_1_nm',
        'low_numbered_approach_hexagons_2_nm',
        'high_numbered_approach_hexagons_2_nm',
        'low_numbered_approach_hexagons_3_nm',
        'high_numbered_approach_hexagons_3_nm',
        'low_numbered_approach_hexagons_4_nm',
        'high_numbered_approach_hexagons_4_nm',
        'low_numbered_approach_hexagons_5_nm',
        'high_numbered_approach_hexagons_5_nm',
        'low_numbered_approach_hexagons_6_nm',
        'high_numbered_approach_hexagons_6_nm',
        'low_numbered_approach_hexagons_7_nm',
        'high_numbered_approach_hexagons_7_nm',
        'low_numbered_approach_hexagons_8_nm',
        'high_numbered_approach_hexagons_8_nm',
        'low_numbered_approach_hexagons_9_nm',
        'high_numbered_approach_hexagons_9_nm',
        'low_numbered_approach_hexagons_10_nm',
        'high_numbered_approach_hexagons_10_nm']

    df = df[id_vars+value_vars]

    # Melting the DataFrame
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                        var_name='gate_id', value_name='hex_id')

    exploded_df = melted_df.explode('hex_id')
    grouping_columns = [col for col in exploded_df.columns if col != 'hex_id']
    deduped_df = exploded_df.drop_duplicates(subset=['hex_id'])
    deduped_df['gate_id_nr'] = deduped_df.gate_id.astype('category').cat.codes*1000
    
    return deduped_df

airport_idents = airports_df['ident'].unique()

failed_apts = []

print(f"Number of airports to process: ", len(airport_idents))

resolution = 11

for apt_icao in airport_idents:
    print("Processing airport: ", apt_icao)
    try: 
      if os.path.exists(f'data/runway_hex/h3_res_{resolution}_rwy_{apt_icao}.parquet'):
        print('AIRPORT EXISTS: ', apt_icao, "-- SKIPPING")
        continue
      
      df = create_hex_airport(apt_icao)
      df.to_parquet(f'data/runway_hex/h3_res_{resolution}_rwy_{apt_icao}.parquet')
      #map_viz = choropleth_map(
      #        df,
      #        column_name='gate_id_nr',
      #        border_color='black',
      #        fill_opacity=0.7,
      #        color_map_name='Reds',
      #        initial_map=None,
      #        initial_location=[df.le_latitude_deg.values[0], df.le_longitude_deg.values[0]],
      #        initial_zoom = 14,
      #        tooltip_columns = ['id', 'airport_ref', 'airport_ident', 'length_ft', 'width_ft',
      #   'surface', 'lighted', 'closed','gate_id']
      #    )
      #map_viz.save(f'data/runway_hex_viz/h3_res_{resolution}_rwy_{apt_icao}.parquet.html')
      print()
    except Exception as e:
      print(f"AIRPORT FAILED {apt_icao}")
      print(f"Error: {e}")
      failed_apts = failed_apts + [apt_icao]
      
print()
print(f"List of failed airports: {failed_apts}")