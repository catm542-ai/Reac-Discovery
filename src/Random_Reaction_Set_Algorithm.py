"""
Random Reaction Set Algorithm
This script systematically generates diverse experimental parameter combinations for Reac-Eval
@author: Cristopher Tinajero
"""

import random
import pandas as pd
import numpy as np

# Step 1: Define the limits and values for input parameters
fr_gas_limits = (50, 500)  # Gas flow rate limits in uL/min
fr_liq_limits = (50, 200)  # Liquid flow rate limits in uL/min
num_gyroid_squares = [2, 4, 6]  # Discrete gyroid configurations
temperature_limits = (60, 120)  # Temperature limits in Celsius
concentration_limits = (2, 4)  # Concentration limits in Molar
wall_thickness_values = np.linspace(125, 300, 3).astype(int)  # Wall thickness values in micrometers

# Step 2: Generate all possible parameter combinations
todas_combinaciones = []
for fr_gas in np.linspace(fr_gas_limits[0], fr_gas_limits[1], 5):
    for fr_liq in np.linspace(fr_liq_limits[0], fr_liq_limits[1], 5):
        for num_gyroid_square in num_gyroid_squares:
            for temperature in range(temperature_limits[0], temperature_limits[1] + 1, 30):
                for concentration in range(concentration_limits[0], concentration_limits[1] + 1):
                    for wall_thickness in wall_thickness_values:
                        experiment = {
                            "fr_gas": fr_gas,
                            "fr_liq": fr_liq,
                            "num_gyroid_squares": num_gyroid_square,
                            "temperature": temperature,
                            "concentration": concentration,
                            "wall_thickness": wall_thickness
                        }
                        todas_combinaciones.append(experiment)

# Step 3: Calculate total combinations and select 3% of them
print("Generating all possible parameter combinations...")
total_combinaciones = len(todas_combinaciones)
uno_por_ciento_combinaciones = int(total_combinaciones * 0.03)

print(f"Total combinations: {total_combinaciones}")
print(f"Selecting 3% of total combinations: {uno_por_ciento_combinaciones} combinations")

# Step 4: Create DataFrame and perform random sampling
todas_combinaciones_df = pd.DataFrame(todas_combinaciones)
combinaciones_seleccionadas_indices = random.sample(range(total_combinaciones), uno_por_ciento_combinaciones)
combinaciones_seleccionadas_df = todas_combinaciones_df.iloc[combinaciones_seleccionadas_indices]

# Step 5: Sort selected combinations by wall thickness and concentration
combinaciones_seleccionadas_df.sort_values(by=["wall_thickness", "concentration"], inplace=True)
print("Randomly selected combinations sorted by wall_thickness and concentration:")
print(combinaciones_seleccionadas_df)

# Step 6: Group and organize data for better readability
grupos_por_wall_thickness = combinaciones_seleccionadas_df.groupby('wall_thickness')
df_anidado = {}
for wall_thickness, grupo_wall in grupos_por_wall_thickness:
    grupos_por_num_gyroid_squares = grupo_wall.groupby('num_gyroid_squares')
    df_anidado[wall_thickness] = {}
    for num_gyroid_squares, grupo_num_gyroid in grupos_por_num_gyroid_squares:
        print(f"Wall thickness: {wall_thickness}, Gyroid squares: {num_gyroid_squares}, Combinations: {len(grupo_num_gyroid)}")
        print(grupo_num_gyroid.head())
        df_anidado[wall_thickness][num_gyroid_squares] = grupo_num_gyroid

# Step 7: Export grouped data to an Excel file
print("Exporting data to Excel...")
with pd.ExcelWriter('resultados_combinaciones.xlsx') as writer:
    for wall_thickness, grupos_num_gyroid in df_anidado.items():
        for num_gyroid_squares, df in grupos_num_gyroid.items():
            sheet_name = f'wall_{wall_thickness}_gyroid_{num_gyroid_squares}'[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Data successfully exported to 'resultados_combinaciones.xlsx'")
