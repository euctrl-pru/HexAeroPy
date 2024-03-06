from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,from_unixtime, min, max, to_date, pandas_udf, col, PandasUDFType, lit, round
from pyspark.sql.types import DoubleType, ArrayType, StructType, StructField, StringType
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
    .appName("OSN ADEP ADES Identification") \
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
    SELECT id, ident, iso_country, continent, latitude_deg, longitude_deg, elevation_ft, type
    FROM {project}.oa_airports
    WHERE (type = 'large_airport' OR type = 'medium_airport');
""")

import math
import json

def generate_circle_polygon(lon, lat, radius_nautical_miles, num_points=360):
    """
    Generate a polygon in GeoJSON format around a given latitude and longitude
    with a specified radius in nautical miles.
    
    :param lat: Latitude of the center point
    :param lon: Longitude of the center point
    :param radius_nautical_miles: Radius in nautical miles
    :param num_points: Number of points to generate for the polygon
    :return: A dictionary representing the polygon in GeoJSON format
    """
    # Convert radius from nautical miles to kilometers
    radius_km = radius_nautical_miles * 1.852
    
    # Function to convert from degrees to radians
    def degrees_to_radians(degrees):
        return degrees * math.pi / 180
    
    # Function to calculate the next point given a distance and bearing
    def calculate_point(lon, lat, distance_km, bearing):
        R = 6371.01  # Earth's radius in kilometers
        lat_rad = degrees_to_radians(lat)
        lon_rad = degrees_to_radians(lon)
        distance_rad = distance_km / R
        bearing_rad = degrees_to_radians(bearing)
        
        lat_new_rad = math.asin(math.sin(lat_rad) * math.cos(distance_rad) +
                                math.cos(lat_rad) * math.sin(distance_rad) * math.cos(bearing_rad))
        lon_new_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_rad) * math.cos(lat_rad),
                                           math.cos(distance_rad) - math.sin(lat_rad) * math.sin(lat_new_rad))
                                           
        lat_new = math.degrees(lat_new_rad)
        lon_new = math.degrees(lon_new_rad)
        return [lon_new, lat_new]
    
    # Generate points
    points = []
    for i in range(num_points):
        bearing = 360 / num_points * i
        point = calculate_point(lon, lat, radius_km, bearing)
        points.append(point)
    points.append(points[0])  # Close the polygon by repeating the first point
    
    # Create GeoJSON
    geojson = {
        "type": "Polygon",
        "coordinates": [points]
    }
    
    geojson_str = json.dumps(geojson)
    
    return geojson_str

def fill_circle_with_hexagons(polygon_json, resolution=8):
    polygon = json.loads(polygon_json)
    hexagons = h3.polyfill(polygon, resolution, geo_json_conformant=True)
    return list(hexagons)

generate_circle_polygon_udf = udf(generate_circle_polygon, StringType())
fill_circle_with_hexagons_udf = udf(fill_circle_with_hexagons, ArrayType(StringType()))

num_points = 360
radia_nm = [5,10,15,20,25,30,35,40,45,50]

pdfs = []
failed_apts = []
for radius_nm in radia_nm:
    print(f"Radius: {radius_nm}")
    for resolution in [5]:
        pdfs = []
        failed_apts = []
        print('Resolution:', resolution)
        for airport in airports_df.toPandas()['ident'].to_list():
            try:
                print('Processing airport: ', airport)
                airports_pdf = airports_df.filter(airports_df.ident == airport).withColumn(
                    "circle_polygon",
                    generate_circle_polygon_udf(
                        airports_df.longitude_deg,
                        airports_df.latitude_deg,
                        lit(radius_nm), 
                        lit(num_points)
                    )
                ).withColumn(
                    "hex_id",
                    fill_circle_with_hexagons_udf(
                        "circle_polygon",
                        lit(resolution) # Resolution
                    )
                ).toPandas()

                airports_pdf = airports_pdf.explode('hex_id')

                airports_pdf['hex_lat'], airports_pdf['hex_lon'] = zip(*airports_pdf['hex_id'].apply(lambda l: h3.h3_to_geo(l)))

                pdfs.append(airports_pdf)
            except Exception as e:
                failed_apts.append(airport)
                print('Airport FAILED:', airport)
                print(f'Error: {e}')

        print(f"Failed airports: {failed_apts}")

        df = pd.concat(pdfs).explode('hex_id')[['id', 'ident', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'type','hex_id', 'hex_lat', 'hex_lon']]

        df.to_parquet(f'data/airport_hex/airport_hex_res_{resolution}_radius_{radius_nm}_nm.parquet')