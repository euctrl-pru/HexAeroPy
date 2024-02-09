import folium
from geojson import Feature, Point, FeatureCollection
import json
import pandas as pd
import matplotlib
import h3

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


def add_trajectory(map_object, dataframe):
    """
    Adds an aircraft trajectory to a Folium map based on coordinates in a Pandas DataFrame.

    Parameters:
    - map_object: Folium Map instance where the trajectory will be added.
    - dataframe: Pandas DataFrame containing the trajectory coordinates with columns 'lat' and 'lon'.
    """
    # Extracting coordinates from DataFrame
    coordinates = dataframe[['lat', 'lon']].values.tolist()
    # Adding a PolyLine to the map to represent the trajectory
    folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(map_object)