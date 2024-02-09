# HexAeroPy

## Introduction

HexAeroPy is a EUROCONTROL Python package designed for aviation professionals and data analysts. It allows for the determination of used airports, runways, taxiways, and stands based on available flight trajectory coordinates. This tool aims to enhance aviation data analysis, facilitating the extraction of milestones for performance analysis.

## Features

-   **Airport Detection**: Identifies airports involved in a flight's trajectory.
-   **Runway Utilization**: Determines which runways are used during takeoff and landing.

## Installation

To install HexAeroPy, ensure you have Python 3.9+ installed. Then run the following command:

``` bash
pip install git+https://github.com/euctrl-pru/hexaeropy.git
```

## Quick Start

Get started with HexAeroPy by running the following Python code:

``` python
from HexAeroPy import *

# Load test data

df = load_dataset(name='trajectories.parquet', datatype='test_data')

# Create unique id for each trajectory

df['id'] = df['icao24'] + '-' + df['callsign'] + '-' + df['time'].dt.date.apply(str)
df = df[['id', 'time', 'icao24', 'callsign', 'lat', 'lon', 'baroaltitude']]

# Identify runways

scored_rwy_detections_df, rwy_detections_df = identify_runways(df)
```

## Usage

### Visualizing Methodology

``` python

# Load approach hex dataset
egll = load_dataset(name="EGLL.parquet", datatype="runway_hex")

# Visualize approach cones
map_viz = choropleth_map(
        egll,
        column_name='gate_id_nr',
        border_color='black',
        fill_opacity=0.7,
        color_map_name='Reds',
        initial_map=None,
        initial_location=[df.lat.values[0], df.lon.values[0]],
        initial_zoom = 13,
        tooltip_columns = ['id', 'airport_ref', 'airport_ident', 'le_ident', 'he_ident', 'length_ft', 'width_ft',
   'surface', 'lighted', 'closed', 'gate_id']
    )

# Add a single aircraft trajectory to the map
df = df[df['id'] == '0a0048-DAH2054-2023-08-02']
add_trajectory(map_viz, df)

# Show
map_viz.save('egll_arrival.html')
```

![Runway detection](assets/egll_departure.png "Departure of a flight of runway 09R/27L at EGLL as detected by HexAeroPy.")

Download the HTML here: [egll_departure.html](assets/egll_departure.html)

## Development Roadmap

-   **[pending implementation] Taxiway Analysis**: Analyzes taxi routes for efficiency and optimization.
-   **[pending implementation] Stand Identification**: Identifies aircraft stands, enhancing ground operation analysis.

## Contributing

We welcome contributions to HexAeroPy! Feel free to submit pull requests, report issues, or request features.

## License

HexAeroPy is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits and Acknowledgments

Special thanks to EUROCONTROL.

## Contact Information

For support or queries, please contact us at [pru-support@eurocontrol.int](mailto:pru-support@eurocontrol.int).
