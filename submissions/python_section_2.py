import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    G = nx.DiGraph()
    
    for index, row in df.iterrows():
        G.add_edge(row[0], row[5], distance=row[1])
    G_undirected = nx.Graph()
    for u, v in G.edges():
        G_undirected.add_edge(u, v, distance=G[u][v][1])
    
    distance_matrix = pd.DataFrame(index=df[0].unique(), columns=df[2].unique())
    for node1 in G_undirected.nodes():
        for node2 in G_undirected.nodes():
            if node1 == node2:
                distance_matrix.loc[node1, node2] = 0
            else:
                try:
                    distance_matrix.loc[node1, node2] = nx.shortest_path_length(G_undirected, node1, node2, weight='distance')
                except nx.NetworkXNoPath:
                    distance_matrix.loc[node1, node2] = float('inf')
    
    
    distance_matrix = distance_matrix.combine_first(distance_matrix.T)
    
    return distance_matrix
    
df = pd.read_csv('dataset-2.csv')
    
distance_matrix = calculate_distance_matrix(df)




def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    df = df.reset_index()
    df = df.rename(columns={'index': 'id_start'})
    
    unrolled_df = pd.melt(df, id_vars='id_start', var_name='id_end', value_name='distance')
    
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    
    return unrolled_df
    
unrolled_distance_matrix = unroll_distance_matrix(distance_matrix)
print(unrolled_distance_matrix)



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    reference_average_distance = df[df['id_start'] == reference_id]['distance'].mean()

    threshold = reference_average_distance * 0.1

    ids_within_threshold = df[(df['distance'] >= reference_average_distance - threshold) &
                              (df['distance'] <= reference_average_distance + threshold) &
                              (df['id_start'] != reference_id)]['id_start'].unique()

    sorted_ids = sorted(ids_within_threshold)

    return sorted_ids


reference_id = 1
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_matrix, reference_id)

print(ids_within_threshold)


def calculate toll_rates(df)->pd.DataFrame():
   """
    Calculate toll rates based on vehicle types.

    Args:
        df (pd.DataFrame): Unrolled distance matrix.

    Returns:
        pd.DataFrame: DataFrame with toll rates for each vehicle type.
    """

    # Wrie your logic here
    rate_coefficients = {

        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    
    return df

df={
    'id_start':[1001400,1001400,1001400,1001400,1001400,1001400,1001400,1001400,1001400,1001400],
    'id_end':[1001402,1001404,1001406,1001408,1001410,1001412,1001414,1001416,1001418,1001420],
    'moto':[7.76,23.92,36.72,54.08,62.96,75.44,90.00,100.56,111.44,121.76],
    'car':[11.64,35.38,55.08,81.12,94.44,113.16,135.00,150.84,167.16,182.64],
    'rv':[14.55,44.85,68.85,101.40,118.05,141.45,168.75,188.55,208.95,228.30],
    'bus':[21.34,65.78,100.98,148.72,173.14,207.46,247.50,276.54,306.46,334.84],
    'truck':[34.92,107.64,165.24,243.36,283.32,339.48,405.00,452.52,501.48,547.92]
}
    
df_with_toll_rates = calculate_toll_rates(df)
    



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    discount_factors = {
        'weekdays': {
            '00:00:00-10:00:00': 0.8,
            '10:00:00-18:00:00': 1.2,
            '18:00:00-23:59:59': 0.8
        },
        'weekends': {
            '00:00:00-23:59:59': 0.7
        }
    }
    
    result_df = pd.DataFrame()
    
    for index, row in df.groupby(['id_start', 'id_end']):
        days = pd.date_range(start='Monday', periods=7, freq='D')
        for day in days:
            if day.weekday() < 5:
                day_type = 'weekdays'
            else:
                day_type = 'weekends'
    
            for time_range, discount_factor in discount_factors[day_type].items():
                start_time, end_time = time_range.split('-')
    
                new_row = {
                    'id_start': index[0],
                    'id_end': index[1],
                    'start_day': day.strftime('%A'), 
                    'start_time': datetime.strptime(start_time, '%H:%M:%S').time(),
                    'end_day': day.strftime('%A'),  # Day of the week (e.g., Monday)
                    'end_time': datetime.strptime(end_time, '%H:%M:%S').time()
                }
    
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    new_row[vehicle] = row[vehicle].values[0] * discount_factor
    
                result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return result_df
    
df ={
    'id_start':[1001400,1001400,1001400,1001400,1001408,1001408,1001408,1001408],
    'id_end':[1001402,1001402,1001402,1001402,1001410,1001410,1001410,1001410],
    'distance':[9.7,9.7,9.7,9.7,11.1,11.1,11.1,11.1]
    'start_day':['Monday','Tuesday','Wednesday','Saturday','Monday,'Tuesday','Wednesday','Saturday'],
    'start_time':[00:00:00, 10:00:00, 18:00:00, 00:00:00,00:00:00,10:00:00,18:00:00,00:00:00],
    'end_day':['Friday', 'Saturday', 'Sunday','Sunday','Friday','Saturday','Sunday','Sunday'],
    'end_time':[10:00:00,18:00:00,23:59:59, 23:59:59,10:00:00,18:00:00,23:59:59, 23:59:59],
    'moto':[6.21,9.31,6.21,5.43,7.10,10.66,7.10,6.22],
    'car':[9.31,13.97,9.31,8.15,10.66,15.98,10.66,9.32],
    'rv':[11.64,17.46,11.64,10.19,13.32,19.98,29.30,19.54,17.09],
    'bus':[17.07,25.61,17.07,14.94,19.54,29.30,19.54,17.09],
    'truck':[27.94,41.90,27.94,24.44,31.97,47.95,31.97,27.97]
}
    
result_df = calculate_time_based_toll_rates(df)
    
    

