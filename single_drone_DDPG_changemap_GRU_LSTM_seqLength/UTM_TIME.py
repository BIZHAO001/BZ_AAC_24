# -*- coding: utf-8 -*-
"""
@Time    : 5/31/2024 4:55 PM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
# Example code execution
import pandas as pd
import pytz
from datetime import datetime

# Example UTC timestamp
utc_timestamp = 1711900799

# Convert the timestamp to a datetime object in UTC
utc_time = pd.to_datetime(utc_timestamp, unit='s', utc=True)

# Define the Singapore timezone
singapore = pytz.timezone('Asia/Singapore')

# Convert the UTC time to Singapore time
singapore_time = utc_time.tz_convert(singapore)

print("UTC Time:", utc_time)
print("Singapore Time:", singapore_time)

# Example filenames
filenames = ["04_01_18_41_08"]


def convert_to_utc_timestamp(filename):
    # Parse the filename
    month, day, hour, minute, second = map(int, filename.split('_'))

    # Create a datetime object (assuming the year is 2024)
    dt = datetime(2024, month, day, hour, minute, second)

    # Convert to UTC timestamp
    utc_timestamp = int(dt.timestamp())

    return utc_timestamp


# Convert all filenames to UTC timestamps
utc_timestamps = [convert_to_utc_timestamp(filename) for filename in filenames]
print(utc_timestamps)