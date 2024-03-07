import pandas as pd
import pkg_resources
import h3pandas
from datetime import datetime
import logging

# Limitation: The current id should represent a flight from ADEP to ADES. If the ID does not represent this, max score vote would mess up the result.
# Solution: Create a new ID which checks whether the flight is one flight, otherwise it would detect and split the id in multiple ids.  

# Limitation: For each track there is not necessarily and airport found
# Solution: Work with larger Radius or Height in airport detection?

#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

def load_dataset(name, datatype):
    """Load a parquet dataset from the package's data directory.
    
    Parameters:
    name (str): The filename of the dataset to load.
    datatype (str): The datatype of file is either 'runway_hex', 'airport_hex' or 'test_data'.

    Returns:
    DataFrame: A pandas DataFrame containing the dataset.
    """
    resource_path = '/'.join(('data', datatype, name))
    file_path = pkg_resources.resource_filename('HexAeroPy', resource_path)
    return pd.read_parquet(file_path)

# Add hex_ids 
def add_statevector_id(df):
    """
    Create a numeric ID for each statevector by using the row index + 1 (to start from 1 instead of 0)
    """
    df['statevector_id'] = range(1, len(df) + 1)
    return df


def add_hex_ids(df, longitude_col='lon', latitude_col ='lat', resolutions=[5, 11]):
    """
    Adds hexagonal IDs to the DataFrame for specified resolutions.
    """
    
    for res in resolutions:
        df = df.h3.geo_to_h3(res, lat_col = latitude_col, lng_col = longitude_col, set_index=False)
        df = df.rename({f'h3_{"{:02d}".format(res)}':f'hex_id_{res}'}, axis=1)
    return df

# Convert altitudes to ft and FL
def convert_baroalt_in_m_to_ft_and_FL(df, baroaltitude_col = 'baroaltitude'):
    """
    Converts barometric altitudes (in meter) to feet (ft) and flight levels (FL).
    """
    df['baroaltitude_ft'] = df[baroaltitude_col] * 3.28084
    df['baroaltitude_fl'] = df['baroaltitude_ft'] / 100
    return df

# Filter out low altitude statevectors

def filter_low_altitude_statevectors(df, baroalt_ft_col = 'baroaltitude_ft', threshold=5000):
    """
    Filters out aircraft states below a specified altitude threshold.
    """
    return df[df[baroalt_ft_col] < threshold]

# Read airport_hexagonifications

def identify_potential_airports(df, track_id_col = 'id', hex_id_col='hex_id', apt_types = ['large_airport', 'medium_airport']):
    """
    Merges aircraft states with airport data based on hex ID (resolution 5).
    """
    airports_df = load_dataset(name = 'airport_hex_res_5_radius_15_nm.parquet', datatype = 'airport_hex')
    airports_df = airports_df[airports_df['type'].isin(apt_types)]
    
    airports_df = airports_df.rename({'id':'apt_id'},axis=1)
    
    # Create list of possible arrival / departure airports
    arr_dep_apt = df.merge(airports_df, left_on='hex_id_5', right_on=hex_id_col, how='left')
    
    # Convert the 'time' column to datetime format if it's not already
    arr_dep_apt['time'] = pd.to_datetime(arr_dep_apt['time'])

    # Initialize the 'segment_status' column with an empty string
    arr_dep_apt['segment_status'] = ''

    # Group by 'id_x' and 'ident'
    grouped = arr_dep_apt.groupby([track_id_col, 'ident'])

    # For each group, find the index of the min and max time and assign 'start' and 'end' respectively
    for name, group in grouped:
        start_index = group['time'].idxmin()
        end_index = group['time'].idxmax()

        # Assign 'start' to the row with the minimum time
        arr_dep_apt.at[start_index, 'segment_status'] = 'start'

        # Assign 'end' to the row with the maximum time
        arr_dep_apt.at[end_index, 'segment_status'] = 'end'

    # Step 1: Filter to only include 'start' or 'end'
    filtered_df = arr_dep_apt[arr_dep_apt['segment_status'].isin(['start', 'end'])]

    # Create separate DataFrames for 'start' and 'end'
    start_df = filtered_df[filtered_df['segment_status'] == 'start'].drop('segment_status', axis=1)
    end_df = filtered_df[filtered_df['segment_status'] == 'end'].drop('segment_status', axis=1)

    # Rename columns to prepend 'start_' or 'end_'
    start_df.columns = ['start_' + col if col not in [track_id_col, 'ident'] else col for col in start_df.columns]
    end_df.columns = ['end_' + col if col not in [track_id_col, 'ident'] else col for col in end_df.columns]

    # Merge the start and end DataFrames on 'id_x' and 'ident'
    apt_detections_df = pd.merge(start_df, end_df, on=[track_id_col, 'ident'], how='outer')

    core = [track_id_col, 'ident', 'start_time',  'start_statevector_id', 'end_time', 'end_statevector_id']
    apt_detections_df = apt_detections_df[core]

    return apt_detections_df


def identify_runways_from_low_trajectories(apt_detections_df, df_f_low_alt):

    # Step 0: Creation of an ID & renaming cols
    apt_detections_df = apt_detections_df.reset_index()
    apt_detections_df['apt_detection_id'] = apt_detections_df['id'] + '_' + apt_detections_df['index'].apply(str)
    apt_detections_df = apt_detections_df[['id', 'ident', 'start_time', 'end_time', 'apt_detection_id']]
    apt_detections_df.columns = ['id', 'apt_det_ident', 'apt_det_start_time', 'apt_det_end_time', 'apt_det_id']

    # Step 1: Convert datetime columns to datetime format if they are not already
    apt_detections_df['apt_det_start_time'] = pd.to_datetime(apt_detections_df['apt_det_start_time'])
    apt_detections_df['apt_det_end_time'] = pd.to_datetime(apt_detections_df['apt_det_end_time'])
    #df_f_low_alt['time'] = pd.to_datetime(df_f_low_alt['time'])

    # Step 2: Merge the data frames on 'id'
    merged_df = pd.merge(df_f_low_alt, apt_detections_df, on='id', how='inner')

    # Step 3: Filter rows where 'time' is between 'apt_det_start_time' and 'apt_det_end_time'
    result_df = merged_df[(merged_df['time'] >= merged_df['apt_det_start_time']) & 
                          (merged_df['time'] <= merged_df['apt_det_end_time'])]
    
    # Step 5: Match with runways

    def match_runways_to_hex(df_low, apt_det_id, apt):

        df_single = df_low[df_low['apt_det_id'] == apt_det_id]

        core_cols_single = ['apt_det_id', 'id', 'time', 'lat', 'lon', 'hex_id_11', 'baroaltitude_fl']

        df_single = df_single[core_cols_single]
        df_single = df_single.reset_index()

        core_cols_rwy = ['id', 'airport_ref', 'airport_ident', 'gate_id', 'hex_id', 'gate_id_nr','le_ident','he_ident']
        try:
            df_rwys = load_dataset(name = f'h3_res_11_rwy_{apt}.parquet', datatype = 'runway_hex')
            df_rwys = df_rwys[core_cols_rwy]

            df_hex_rwy = df_single.merge(df_rwys,left_on='hex_id_11', right_on='hex_id', how='left')

            result = df_hex_rwy.groupby(['apt_det_id', 'id_x','airport_ident', 'gate_id','le_ident','he_ident'])['time'].agg([min,max]).reset_index().sort_values('min')
            return result
        except Exception as e:
            logging.warning(f'Warning: Due to limited data in OurAirports, airport [{apt}] does not have the runway config. No matching for this airport.')
            return pd.DataFrame.from_dict({'id_apt':[], 'airport_ident':[], 'gate_id':[]})

    dfs = apt_detections_df.apply(lambda l: match_runways_to_hex(result_df, l['apt_det_id'],l['apt_det_ident']),axis=1).to_list()

    result = pd.concat(dfs)
    
    # If you detected an aircraft at an airport, it might be the case that for the same ID the aircraft 
    # lands and takes off.. Therefore we now generate different ids if the 
    # aircraft has a break of 40 minutes between detections in any sections.. 
    
    rwy_detections_df = result.rename({'id_x':'id'}, axis=1)[['id', 'apt_det_id', 'airport_ident', 'gate_id', 'le_ident', 'he_ident', 'min', 'max']]

    rwy_id_gen = rwy_detections_df.groupby(['id', 'apt_det_id', 'airport_ident', 'le_ident', 'he_ident'])['min'].agg(min).reset_index()
    rwy_id_gen = rwy_id_gen.sort_values(['id', 'apt_det_id', 'airport_ident', 'min'])

    # Step 1: Convert 'min' column to datetime
    rwy_id_gen['min'] = pd.to_datetime(rwy_id_gen['min'])

    # Step 2 & 3: Calculate time difference within each group
    rwy_id_gen.sort_values(by=['id', 'apt_det_id', 'airport_ident', 'min'], inplace=True)
    rwy_id_gen['time_diff'] = rwy_id_gen.groupby(['id', 'apt_det_id', 'airport_ident'])['min'].diff()

    # Convert time difference to minutes for easier comparison
    rwy_id_gen['time_diff_minutes'] = rwy_id_gen['time_diff'].dt.total_seconds() / 60 

    # Step 4: Generate new 'rwy_det_id' where time difference is greater than 40 minutes
    def generate_rwy_det_id(row, id_counter):
        if row['time_diff_minutes'] > 40:
            id_counter[row['apt_det_id']] += 1
        return f"{row['apt_det_id']}_{id_counter[row['apt_det_id']]}"

    # Initialize a counter for each apt_det_id to append numbers for new ids
    id_counter = {key: 0 for key in rwy_id_gen['apt_det_id'].unique()}
    rwy_id_gen['rwy_det_id'] = rwy_id_gen.apply(generate_rwy_det_id, axis=1, id_counter=id_counter)

    # Optionally, you can drop the intermediate columns if they are no longer needed
    rwy_id_gen.drop(columns=['time_diff', 'time_diff_minutes', 'min'], inplace=True)


    rwy_detections_df = rwy_detections_df.merge(rwy_id_gen, on = ['id', 'apt_det_id', 'airport_ident', 'le_ident', 'he_ident'], how='left')

    rwy_detections_df = rwy_detections_df[['id', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'gate_id', 'le_ident', 'he_ident', 'min', 'max']]

    return rwy_detections_df

def manipulate_df_and_determine_arrival_departure(df):
    result = df.copy()

    def clean_gate(gate_id):
        if gate_id == 'runway_hexagons':
            return 'runway_hexagons',0
        else:
            return '_'.join(gate_id.split('_')[:4]), int(gate_id.split('_')[4])

    result['gate_type'], result['gate_distance_from_rwy_nm'] = zip(*result.gate_id.apply(clean_gate))

    ## Determining arrival / departure... 

    result = result.reset_index(drop=True).rename({'id':'id_x'},axis=1)

    result_min = result.loc[result.groupby(['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident'])['gate_distance_from_rwy_nm'].idxmin()]
    result_max = result.loc[result.groupby(['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident'])['gate_distance_from_rwy_nm'].idxmax()] 

    # Copy the DataFrame to avoid modifying the original unintentionally
    result_copy = result.copy()

    # Compute the minimum and maximum 'gate_distance_from_rwy_nm' for each group
    min_values = result.groupby(['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident'])['gate_distance_from_rwy_nm'].transform('min')
    max_values = result.groupby(['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident'])['gate_distance_from_rwy_nm'].transform('max')

    # Add these as new columns to the DataFrame
    result_copy['min_gate_distance'] = min_values
    result_copy['max_gate_distance'] = max_values

    # Now, you can filter rows where 'gate_distance_from_rwy_nm' matches the min or max values
    # To specifically keep rows with the minimum value:
    result_min = result_copy[result_copy['gate_distance_from_rwy_nm'] == result_copy['min_gate_distance']]

    # To specifically keep rows with the maximum value:
    result_max = result_copy[result_copy['gate_distance_from_rwy_nm'] == result_copy['max_gate_distance']]


    cols_of_interest = ['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident', 'min', 'gate_distance_from_rwy_nm']
    result_min = result_min[cols_of_interest].rename({'min':'time_entry_min_distance', 'gate_distance_from_rwy_nm':'min_gate_distance_from_rwy_nm'},axis=1)
    result_max = result_max[cols_of_interest].rename({'min':'time_entry_max_distance', 'gate_distance_from_rwy_nm':'max_gate_distance_from_rwy_nm'},axis=1)

    det = result_min.merge(result_max, on=['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident'], how='outer')

    det['time_since_minimum_distance'] = det['time_entry_min_distance']-det['time_entry_max_distance']

    det['time_since_minimum_distance_s'] = det['time_since_minimum_distance'].dt.total_seconds()

    det['status'] = det['time_since_minimum_distance_s'].apply(lambda l: 'arrival' if l > 0 else 'departure')
    det['status'] = det['status'].fillna('undetermined')

    det = det[['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident','status']]

    gb_cols = ['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident', 'gate_type']
    result = result.groupby(gb_cols).agg(
        entry_time_approach_area=('min', 'min'),
        exit_time_approach_area=('max', 'max'),
        intersected_subsections=('gate_distance_from_rwy_nm', 'count'),
        minimal_distance_runway=('gate_distance_from_rwy_nm', 'min'),
        maximal_distance_runway=('gate_distance_from_rwy_nm', 'max')
    )
    result = result.reset_index()
    return result, det

def score_and_apply_heuristics(df, det):
    result = df.copy()
    
    rwy_result_cols = ['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident']

    rwy_result = result[rwy_result_cols + ['gate_type']]
    rwy_result = rwy_result[rwy_result['gate_type']=='runway_hexagons']
    rwy_result = rwy_result[rwy_result_cols]
    rwy_result['runway_detected'] = True

    result = result.merge(rwy_result, on=rwy_result_cols, how = 'left')

    result['runway_detected'] = result['runway_detected'].fillna(False)
    
    result['high_number_intersections'] = result['intersected_subsections']>5

    result['minimal_number_intersections'] = result['intersected_subsections']>2
    
    result['low_minimal_distance'] = result['minimal_distance_runway']<5

    result['touched_closest_segment_to_rw'] = result['minimal_distance_runway']==1

    result['touched_second_closest_segment_to_rw'] = result['minimal_distance_runway']<=2 

    approach_detected_weight = 0.3
    rwy_detected_weight = 2
    high_number_intersections_weight = 1 
    low_minimal_distance_weight = 1
    touched_closest_segment_to_rw_weight = 1.5
    touched_second_closest_segment_to_rw_weight = 0.75

    max_score = approach_detected_weight + rwy_detected_weight + high_number_intersections_weight + low_minimal_distance_weight + touched_closest_segment_to_rw_weight + touched_second_closest_segment_to_rw_weight

    result['score'] = result['minimal_number_intersections'].apply(int) * (
                       1*approach_detected_weight + # For all flights in this dataset an approach is detected (i.e., they entered the approach cone)
                       result['runway_detected'].apply(int)*rwy_detected_weight + 
                       result['high_number_intersections'].apply(int)*high_number_intersections_weight + 
                       result['low_minimal_distance'].apply(int)*low_minimal_distance_weight + 
                       result['touched_closest_segment_to_rw'].apply(int)*touched_closest_segment_to_rw_weight + 
                       result['touched_second_closest_segment_to_rw'].apply(int)*touched_second_closest_segment_to_rw_weight
                      ) / max_score * 100
    
    result = result[result['score'] > 10] # Minimal requirement to ensure quality. Otherwise it's just a touch
    
    result = result.reset_index(drop=True)

    result = result.merge(det,on=['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident','le_ident','he_ident'], how ='left')

    result['status'] = result['status'].fillna('undetermined')

    result['rwy'] = result['le_ident'] + '/' + result['he_ident']

    rwy_winner = result.loc[result.groupby(['id_x','apt_det_id', 'rwy_det_id','airport_ident'])['score'].idxmax()].copy()
    rwy_winner['score'] = rwy_winner['score'].apply(str)
    rwy_winner = rwy_winner.groupby(['id_x', 'apt_det_id', 'rwy_det_id', 'airport_ident'])[['le_ident', 'he_ident', 'rwy', 'score', 'status']].agg(', '.join).reset_index()

    rwy_winner = rwy_winner.rename({
        'id_x':'id',
        'rwy' : 'likely_rwy',
        'score': 'likely_rwy_score',
        'status': 'likely_rwy_status'
        }, axis=1)

    id_cols = ['id', 'apt_det_id', 'rwy_det_id', 'airport_ident', 'le_ident', 'he_ident']
    rwy_winner_flag = rwy_winner[id_cols].copy()

    rwy_winner_flag.loc[:,'winner'] = True

    result = result.rename({'id_x':'id'}, axis=1)
    result = result.merge(rwy_winner_flag, on = id_cols, how='left') 
    result['winner'] = result['winner'].fillna(False)

    rwy_losers = result[result['winner']==False].copy()

    rwy_losers['score'] = rwy_losers.loc[:,'score'].apply(str)
    rwy_losers = rwy_losers.groupby(['id', 'apt_det_id', 'rwy_det_id','airport_ident'])[['le_ident', 'he_ident', 'rwy','score', 'status']].agg(', '.join).reset_index()


    rwy_losers = rwy_losers.rename({
        'rwy' : 'potential_other_rwys',
        'score': 'potential_other_rwy_scores',
        'status': 'potential_other_rwy_status'
        }, axis=1)[['id','apt_det_id', 'airport_ident', 'potential_other_rwys', 'potential_other_rwy_scores', 'potential_other_rwy_status']]

    rwy_determined = rwy_winner.merge(rwy_losers, on=['id','apt_det_id','airport_ident'], how='left')

    return rwy_determined


def identify_runways(df, track_id_col = 'id', longitude_col = 'lon', latitude_col = 'lat', baroaltitude_col = 'baroaltitude'):
    
    print('[HexAero for Python - Starting engines...]')
    
    print(f'[STAGE 1] Reading statevectors... ({datetime.now()})')
    df_w_id = add_statevector_id(df)
    
    print(f'[STAGE 2] Adding hex ids... ({datetime.now()})')
    df_w_hex = add_hex_ids(df_w_id, longitude_col=longitude_col, latitude_col=latitude_col,  resolutions=[5, 11])
    
    print(f'[STAGE 3] Converting baroaltitudes to FL... ({datetime.now()})')
    df_w_baroalt_ft_fl = convert_baroalt_in_m_to_ft_and_FL(df_w_hex, baroaltitude_col = baroaltitude_col)
    
    print(f'[STAGE 4] Filtering low altitudes for airport matching... ({datetime.now()})')
    df_f_low_alt = filter_low_altitude_statevectors(df_w_baroalt_ft_fl, baroalt_ft_col = 'baroaltitude_ft', threshold=5000)
    
    print(f'[STAGE 5] Finding matching airports... ({datetime.now()})')
    apt_detections_df = identify_potential_airports(df_f_low_alt, track_id_col = track_id_col, hex_id_col='hex_id', apt_types = ['large_airport', 'medium_airport'])
    
    print(f'[STAGE 6] Finding matching runways... ({datetime.now()})')
    rwy_detections_df = identify_runways_from_low_trajectories(apt_detections_df,df_f_low_alt)
    
    print(f'[STAGE 7] Applying heuristics to determine most likely runways... ({datetime.now()})')
    rwy_detections_df, det = manipulate_df_and_determine_arrival_departure(rwy_detections_df)
    scored_rwy_detections_df = score_and_apply_heuristics(rwy_detections_df, det)
    
    print(f'[DONE] Thank you for flying with HexAero... ({datetime.now()})')
         
    return scored_rwy_detections_df, rwy_detections_df
    
# Trajectories
#df = pd.read_parquet('../data/2023-08-02-11.parquet')
#df['id'] = df['icao24'] + '-' + df['callsign'] + '-' + df['time'].dt.date.apply(str)
#df = df[['id', 'time', 'icao24', 'callsign', 'lat', 'lon', 'baroaltitude']]

#scored_rwy_detections_df, rwy_detections_df = identify_runways(df)
