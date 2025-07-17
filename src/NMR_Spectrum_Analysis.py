"""
NMR Spectrum Analysis: Automatic Reading and Peak Identification
@author: Cristopher Tinajero
"""

import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 1. Capture the current date for directory organization
def capture_date():
    """
    Captures the current date and formats it into day, month, and year strings.

    Returns:
        tuple: A tuple containing day, month, and year strings.
    """
    current_date = datetime.now()
    day = str(current_date.day).zfill(2)
    month = str(current_date.month).zfill(2)
    year = str(current_date.year)
    return day, month, year

# Capture current date
day_str, month_str, year_str = capture_date()
print(f"Day: {day_str}, Month: {month_str}, Year: {year_str}")

# 2. Get the latest folder in the specified directory
def get_latest_folder(directory):
    """
    Finds the most recently created folder in a given directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        str: The name of the most recent folder or None if no folders are found.
    """
    try:
        folders = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
        if not folders:
            print("No folders found in the directory.")
            return None
        folders.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True)
        return folders[0]
    except FileNotFoundError:
        print("Directory not found.")
        return None

# Directory setup
base_directory = f'C:\\Users\\tinajero\\Desktop\\DATA - Acceso directo\\{year_str}\\{month_str}\\{day_str}'
latest_folder_name = get_latest_folder(base_directory)

if latest_folder_name:
    folder_path = os.path.join(base_directory, latest_folder_name)
    print(f"Latest folder: {folder_path}")
else:
    folder_path = None

# 3. Read spectrum data from a CSV file
def read_spectrum_data(folder_path):
    """
    Reads spectral data from a CSV file within a specified folder.

    Args:
        folder_path (str): Path to the folder containing the CSV file.

    Returns:
        list: A list of rows read from the CSV file.
    """
    file_name = "spectrum_processed.csv"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.isfile(file_path):
        print(f"File {file_name} not found in folder {folder_path}.")
        return []

    data = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            data = [row for row in csv_reader]
    except IOError:
        print(f"Could not open file: {file_path}")
    return data

# Load spectrum data
if folder_path:
    data = read_spectrum_data(folder_path)

# Process spectrum data
x, y = [], []
for i in range(1, len(data)):
    try:
        x1, x2 = data[i][0].split('.')
        y1, y2 = data[i][1].split(';')
        x_value = float(f"{x1}.{x2}")
        y_value = float(f"{y1}.{y2}")
        if 0 <= x_value <= 5:  # Filter x range
            x.append(x_value)
            y.append(y_value)
    except ValueError:
        continue

# 4. Identify peaks in the spectrum
def find_peaks(x, y, threshold):
    """
    Identifies peaks in the spectrum based on a threshold value.

    Args:
        x (list): List of x-values (chemical shifts).
        y (list): List of y-values (intensities).
        threshold (float): Minimum y-value to identify a peak.

    Returns:
        list: List of peaks with their start and end indices.
    """
    peaks = []
    start = None
    peak_num = 1

    for i in range(len(y)):
        if y[i] >= threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                peaks.append((peak_num, start, end))
                start = None
                peak_num += 1
    if start is not None:
        end = len(y) - 1
        peaks.append((peak_num, start, end))
    return peaks

# Calculate peak areas
def calculate_area(x, y, start, end):
    """
    Calculates the area under a peak using the trapezoidal rule.

    Args:
        x (list): List of x-values.
        y (list): List of y-values.
        start (int): Start index of the peak.
        end (int): End index of the peak.

    Returns:
        float: Area under the peak.
    """
    area = sum((x[i+1] - x[i]) * (y[i] + y[i+1]) / 2 for i in range(start, end))
    return area

# Define threshold and find peaks
threshold = 4.5
peaks = find_peaks(x, y, threshold)

# Calculate areas and proportions
areas = [calculate_area(x, y, start, end) for _, start, end in peaks]
total_area = sum(areas)
proportions = [area / total_area for area in areas]

# Create a DataFrame for results
df = pd.DataFrame({
    'Peak Number': [peak[0] for peak in peaks],
    'Start x': [x[peak[1]] for peak in peaks],
    'End x': [x[peak[2]] for peak in peaks],
    'Area': areas,
    'Proportion': proportions
})

# Save results to Excel
df.to_excel('peak_results.xlsx', index=False)

# Plot the spectrum with identified peaks
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Spectrum")
for peak, start, end in peaks:
    plt.axvline(x[start], color='green', linestyle='--', linewidth=0.8)
    plt.axvline(x[end], color='red', linestyle='--', linewidth=0.8)
    plt.text(x[start], y[start], str(peak), verticalalignment='bottom', fontsize=8)
plt.xlabel('Chemical Shift (ppm)')
plt.ylabel('Intensity')
plt.title('NMR Spectrum with Peaks')
plt.legend()
plt.grid(True)
plt.show()
