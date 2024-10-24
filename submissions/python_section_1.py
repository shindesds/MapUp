from typing import Dict, List

import pandas as pd
import re
from datetime import datetime
import polyline
import math

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.

    """
    # Your code goes here.

    left=0
    right=n-1
    while start < end:
        temp=list[left]
        list[left]=list[right]
        list[right]=temp

        left=left+1
        right=right-1
    return lst
print(reverse_by_n_elements(lst))   
list=[2, 4, 5, 6, 75, 89, 56, 76, 98]
n=3

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """

    dict = {}

    for string in lst:
        length = len(string)

        if length not in dict:
            dict[length] = []


        dict[length].append(string)

    return dict

strings=["apple", "banana", "Cherray", "cat", "bat"]
print(group_by_length(lst))



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here:

    flattened_dict = {}

    for key, value in nested_dict.items():
        new_key = f"{prefix}{sepr}{key}" if prefix else key

        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key, sep))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    flattened_dict.update(flatten_dict(item, f"{new_key}_{index}", sep))
                else:
                    flattened_dict[f"{new_key}_{index}"] = item
        else:

            flattened_dict[new_key] = value

    return flattened_dict

nested_dict={
    "name": "John Doe",
    "age": 30,
    "address": {
        "street": "123 Main St",
        "city": "Mumbai",
        "state": "Maharashtra",
        "zip": "411012"
    },
    "interests": ["reading", "hiking", "coding"],
    "education": {
        "high_school": "Mumbai High",
        "college": "Shivaji University",
        "degrees": ["BA", "MS"]
    }
}

flatten_dict=flatten_dict(nested_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
def permute_unique(nums):
    def backtrack(l, r):
        if l == r:
            permutations.add(tuple(nums))
        for i in range(l, r):
            nums[l], nums[i] = nums[i], nums[l]
            backtrack(l + 1, r)
            nums[l], nums[i] = nums[i], nums[l]

    nums.sort()
    permutations = set()
    backtrack(0, len(nums))
    return [list(p) for p in permutations]
    pass

nums=[2,3,3]



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    patterns = {
        r'\b(\d{2})-(\d{2})-(\d{4})\b': '%d-%m-%Y',  # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b': '%d/%m/%Y',  # dd/mm/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b': '%Y.%m.%d'  # yyyy.mm.dd
    }

    valid_dates = []

    for pattern, date_format in patterns.items():
        matches = re.findall(pattern, text)

        
        for match in matches:
            try:
                date_string = '-'.join(match) if date_format == '%d-%m-%Y' else '/'.join(match) if date_format == '%d/%m/%Y' else '.'.join(match)
                datetime.strptime(date_string, date_format)
                valid_dates.append(date_string)
            except ValueError:
                pass

    return valid_dates
    pass
text= "My birthdays are 01-01-1990, 01/01/1995, and 1990.01.01 "


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """

    coordinates = polyline.decode(polyline_str)
    
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000
    
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
    
        return distance
    
    df['distance'] = df.apply(lambda row: haversine_distance(row['latitude'], row['longitude'],
                                                                 df.iloc[0]['latitude'], df.iloc[0]['longitude'])
                            if row.name == 0 else
                        haversine_distance(row['latitude'], row['longitude'],
                                        df.iloc[row.name-1]['latitude'], df.iloc[row.name-1]['longitude']),
                            axis=1)
    

    return pd.Dataframe()
    
polyline_str = "u xw|Ey qVt C"



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here

    n = len(matrix)

    rotated_matrix = [list(reversed(col)) for col in zip(*matrix)]

    transformed_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i][:j] + rotated_matrix[i][j+1:])
            col_sum = sum(rotated_matrix[k][j] for k in range(n) if k != i)
            transformed_matrix[i][j] = row_sum + col_sum

    return transformed_matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]



def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])
    
    df['startHour'] = df['startTime'].dt.hour
    df['endHour'] = df['endTime'].dt.hour
    df['startDayOfWeek'] = df['startTime'].dt.dayofweek  
    
    def check_completeness(group):
            
        hours = set(range(24))  
        group_hours = set(group['startHour']) | set(group['endHour'])
        if not hours.issubset(group_hours):
            return True  
    
            
        days = set(range(7)) 
        group_days = set(group['startDayOfWeek'])
        if not days.issubset(group_days):
            return True  
    
        return False 

    return pd.Series()

    
df = pd.read_csv('dataset-2.csv')
    
incorrect_timestamps = time_check(df)




    
