!pip install h3
!pip install h3_pyspark
!pip install geopandas

import geopandas as gpd
import pandas as pd
import h3

#from pyspark.sql import SparkSession, Window
#from pyspark.sql import functions as F
#from pyspark.sql.functions import lit
#from pyspark.sql.window import Window
#from pyspark.sql import DataFrame
import shutil
from datetime import datetime
#import h3_pyspark

# Settings
project = "project_aiu"
resolutions = [7,8,9,10]

for resolution in resolutions:
    print(f"Working on resolution: {resolution}")
    # Getting today's date
    today = datetime.today().strftime('%d %B %Y')

    # Spark Session Initialization
    #shutil.copy("/runtime-addons/cmladdon-2.0.40-b150/log4j.properties", "/etc/spark/conf/") # Setting logging properties
    #spark = SparkSession.builder \
    #    .appName("OSN tracks ETL") \
    #    .config("spark.log.level", "ERROR")\
    #    .config("spark.ui.showConsoleProgress", "false")\
    #    .config("spark.hadoop.fs.azure.ext.cab.required.group", "eur-app-aiu-dev") \
    #    .config("spark.kerberos.access.hadoopFileSystems", "abfs://storage-fs@cdpdldev0.dfs.core.windows.net/data/project/aiu.db/unmanaged") \
    #    .config("spark.driver.cores", "1") \
    #    .config("spark.driver.memory", "10G") \
    #    .config("spark.executor.memory", "8G") \
    #    .config("spark.executor.cores", "1") \
    #    .config("spark.executor.instances", "2") \
    #    .config("spark.dynamicAllocation.maxExecutors", "5") \
    #    .config("spark.network.timeout", "800s") \
    #    .config("spark.executor.heartbeatInterval", "400s") \
    #    .enableHiveSupport() \
    #    .getOrCreate()


    # Settings 

    # Read the shapefile
    file_path = 'assets/firs_nm_406/firs_nm_406.shp'
    gdf = gpd.read_file(file_path)

    # Filter out some FIRs which behave oddly - To be examined
    gdf=gdf.reset_index().rename({'index':'ROW_NR'},axis=1)
    f = gdf['ROW_NR'].isin([4, 14, 37, 39, 68, 72, 80, 93, 135]) # Filter out troublesome FIRs... to be solved later :)
    gdf = gdf[~f]


    # Create the H3 tags for each polygon in the shapefile 
    hex_ids = []

    # Loop through each row in the GeoDataFrame
    for index, row in gdf.iterrows():
        # Convert geometry to geo interface format
        geo_interface = row['geometry'].__geo_interface__

        # Reformat the coordinates for the h3.polyfill function
        if len(geo_interface['coordinates']) == 1:
            formatted_coordinates = [[[y, x] for x, y, _ in list(geo_interface['coordinates'][0])]]
            polygon = {'type': 'Polygon', 'coordinates': formatted_coordinates}
        else: 
            print(index)

        hexagons = h3.polyfill(polygon, res = resolution)
        hexagons = h3.compact(hexagons)

        # Append the current DataFrame to the result DataFrame
        hex_ids = hex_ids + [list(hexagons)]

    # Assign HEX_IDs and explode for 1 row per HEX_ID
    gdf['HEX_ID'] = hex_ids
    gdf = gdf[['AC_ID', 'AV_AIRSPAC', 'AV_ICAO_ST', 'MIN_FLIGHT', 'MAX_FLIGHT', 'AV_NAME', 'OBJECTID', 'SHAPE_AREA', 'SHAPE_LEN', 'HEX_ID']].explode('HEX_ID')

    # Add EUROCONTROL Network Manager FIR flag
    eurocontrol_nm_icao_codes = [
        "EB", # Belgium
        "LF", # France
        "ED", "ET", # Germany
        "EL", # Luxembourg
        "EH", # The Netherlands
        "EG", # United Kingdom
        "EI", # Ireland
        "LP", # Portugal
        "LG", # Greece
        "LM", # Malta
        "LT", # Türkiye
        "LC", # Cyprus
        "LH", # Hungary
        "LS", # Switzerland
        "LO", # Austria
        "EK", # Denmark
        "EN", # Norway
        "LJ", # Slovenia
        "ES", # Sweden
        "LK", # Czech Republic
        "LI", # Italy
        "LR", # Romania
        "LB", # Bulgaria
        "LD", # Croatia
        "LN", # Monaco
        "LZ", # Slovakia
        "LE", "GC", "GE", # Spain
        "LW", # North Macedonia
        "LU", # Republic of Moldova
        "EF", # Finland
        "LA", # Albania
        "LQ", # Bosnia and Herzegovina
        "EP", # Poland
        "UK", # Ukraine
        "LY", # Serbia
        "UD", # Armenia
        "EY", # Lithuania
        "LY", # Montenegro
        "EV", # Latvia
        "UG", # Georgia
        "EE", # Estonia
        "LL", # Israel
        "GM", "GO", # Morocco
    ]

    gdf['NETWORK_MANAGER_AREA'] = gdf['AV_ICAO_ST'].isin(eurocontrol_nm_icao_codes)

    # Add resolution of each cell
    gdf['HEX_ID_RESOLUTION'] = gdf['HEX_ID'].apply(lambda l: h3.h3_get_resolution(l))

    # Add maximal resolution used 
    gdf['HEX_ID_MAX_RESOLUTION'] = resolution

    # Add center coordinates of each cell 
    gdf['HEX_ID_CENTER_LAT'], gdf['HEX_ID_CENTER_LON'] = zip(*gdf['HEX_ID'].apply(lambda l:h3.h3_to_geo(l)))

    gdf = gdf[[
        'AC_ID',
        'AV_AIRSPAC',
        'AV_ICAO_ST',
        'MIN_FLIGHT',
        'MAX_FLIGHT',
        'AV_NAME',
        'OBJECTID',
        'SHAPE_AREA',
        'SHAPE_LEN',
        'HEX_ID',
        'HEX_ID_RESOLUTION', 
        'HEX_ID_CENTER_LAT',
        'HEX_ID_CENTER_LON',
        'HEX_ID_MAX_RESOLUTION',
        'NETWORK_MANAGER_AREA']]

    gdf.to_parquet(f'data/fir_hex/h3_min_res_{resolution}_fir.parquet')

#spark_df = spark.createDataFrame(gdf)

#project = 'project_aiu'
#create_fir_h3_binning = f"""
#CREATE TABLE IF NOT EXISTS `{project}`.`fir_h3_binning_unclustered` (
#    AC_ID INT COMMENT 'This column represents the identifier for the airspace sector or a specific code related to air traffic control (unclear).',
#    AV_AIRSPAC STRING COMMENT 'This column represents the name of the airspace, such as a specific Flight Information Region (FIR).',
#    AV_ICAO_ST STRING COMMENT 'This is the ICAO standard code for the country in which the FIR is located.',
#    MIN_FLIGHT INT COMMENT 'This column denotes the minimum flight level applicable within the FIR.',
#    MAX_FLIGHT INT COMMENT 'This column denotes the maximum flight level applicable within the FIR.',
#    AV_NAME STRING COMMENT 'This is the name of the airspace or FIR.',
#    OBJECTID INT COMMENT 'A unique identifier for each FIR.',
#    SHAPE_AREA DOUBLE COMMENT 'This represents the total area of the FIR.',
#    SHAPE_LEN DOUBLE COMMENT 'This indicates the perimeter length of the FIR or sector.',
#    HEX_ID STRING COMMENT 'A H3 hexadecimal identifier, for a specific h3 hexagon within the FIR.',
#    HEX_ID_RESOLUTION INT COMMENT 'Indicates the H3 resolution of the HEX_ID.',
#    HEX_ID_CENTER_LAT DOUBLE COMMENT 'Geographical latitude of the center point of the H3 hexagon defined by the HEX_ID.',
#    HEX_ID_CENTER_LON DOUBLE COMMENT 'Geographical longitude of the center point of the H3 hexagon defined by the HEX_ID.',
#    HEX_ID_MAX_RESOLUTION INT COMMENT 'The maximum H3 resolution level for any of the HEX_ID.',
#    NETWORK_MANAGER_AREA BOOLEAN COMMENT 'Indicates whether the specific FIR or sector is part of the EUROCONTROL Network Manager airspace.'
#)
#COMMENT 'This table contains detailed information about various Flight Information Regions extracted from https://github.com/euctrl-pru/pruatlas/blob/master/data-raw/firs_nm_406.zip shapefiles. Each FIR shape is filled with compact H3 hexagons with a maximal resolution.'
#--CLUSTERED BY (AV_NAME, HEX_ID, HEX_ID_RESOLUTION, HEX_ID_CENTER_LAT, HEX_ID_CENTER_LON, HEX_ID_MAX_RESOLUTION) INTO 4096 BUCKETS
#STORED AS parquet
#TBLPROPERTIES ('transactional'='false');
#"""

#spark.sql(f"DROP TABLE IF EXISTS `{project}`.`fir_h3_binning_unclustered`;")
#spark.sql(create_fir_h3_binning)

#spark_df.write.mode("append").insertInto(f"`{project}`.`fir_h3_binning_unclustered`")

## Clustering
#create_fir_h3_binning_clustered = f"""
#CREATE TABLE IF NOT EXISTS `{project}`.`fir_h3_binning` (
#    AC_ID INT COMMENT 'This column represents the identifier for the airspace sector or a specific code related to air traffic control (unclear).',
#    AV_AIRSPAC STRING COMMENT 'This column represents the name of the airspace, such as a specific Flight Information Region (FIR).',
#    AV_ICAO_ST STRING COMMENT 'This is the ICAO standard code for the country in which the FIR is located.',
#    MIN_FLIGHT INT COMMENT 'This column denotes the minimum flight level applicable within the FIR.',
#    MAX_FLIGHT INT COMMENT 'This column denotes the maximum flight level applicable within the FIR.',
#    AV_NAME STRING COMMENT 'This is the name of the airspace or FIR.',
#    OBJECTID INT COMMENT 'A unique identifier for each FIR.',
#    SHAPE_AREA DOUBLE COMMENT 'This represents the total area of the FIR.',
#    SHAPE_LEN DOUBLE COMMENT 'This indicates the perimeter length of the FIR or sector.',
#    HEX_ID STRING COMMENT 'A H3 hexadecimal identifier, for a specific h3 hexagon within the FIR.',
#    HEX_ID_RESOLUTION INT COMMENT 'Indicates the H3 resolution of the HEX_ID.',
#    HEX_ID_CENTER_LAT DOUBLE COMMENT 'Geographical latitude of the center point of the H3 hexagon defined by the HEX_ID.',
#    HEX_ID_CENTER_LON DOUBLE COMMENT 'Geographical longitude of the center point of the H3 hexagon defined by the HEX_ID.',
#    HEX_ID_MAX_RESOLUTION INT COMMENT 'The maximum H3 resolution level for any of the HEX_ID.',
#    NETWORK_MANAGER_AREA BOOLEAN COMMENT 'Indicates whether the specific FIR or sector is part of the EUROCONTROL Network Manager airspace.'
#)
#COMMENT 'This table contains detailed information about various Flight Information Regions extracted from https://github.com/euctrl-pru/pruatlas/blob/master/data-raw/firs_nm_406.zip shapefiles. Each FIR shape is filled with compact H3 hexagons with a maximal resolution.'
#--CLUSTERED BY (AV_NAME, HEX_ID, HEX_ID_RESOLUTION, HEX_ID_CENTER_LAT, HEX_ID_CENTER_LON, HEX_ID_MAX_RESOLUTION) INTO 4096 BUCKETS
#STORED AS parquet
#TBLPROPERTIES ('transactional'='false');
#"""

#spark.sql(f"DROP TABLE IF EXISTS `{project}`.`fir_h3_binning`;")
#spark.sql(create_fir_h3_binning_clustered)

#spark.sql(f"""
#        INSERT INTO TABLE `{project}`.`fir_h3_binning`
#        SELECT * FROM `{project}`.`fir_h3_binning_unclustered`""")

#spark.sql(f"DROP TABLE IF EXISTS `{project}`.`fir_h3_binning_unclustered`;")