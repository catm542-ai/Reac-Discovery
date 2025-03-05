# Reac-Discovery
**Description:**
Reac-Discovery is a methodology composed of multiple Python scripts, each with a specific function, designed for the generation, fabrication, and optimization of catalytic reactors in continuous flow. This repository does not automatically execute the software; instead, each part must be used sequentially to achieve the desired optimization. Below, the workflow and function of each script are detailed.
**Table of Contents:**
System Requirements
Installation
Workflow
Project Structure
Contributions
License
**System requirements:**
**Dependencies and Versions:**
These are the dependencies used in this project:
+ Scientific Computing and Data Processing
numpy==2.1.0
pandas==2.2.3
scipy==1.11.1
matplotlib==3.7.1
seaborn==0.12.2
openpyxl==3.1.2
itertools (Built-in Python module)
+ Machine Learning and Data Processing
tensorflow==2.12.0
keras==2.12.0
scikit-learn==0.24.2
joblib==1.3.2
+ 3D Geometry and Mesh Processing
trimesh==4.5.1
pyglet==2.0.10
pyopengl==3.1.6
+ Reaction Automation and Hardware Control
rseriesopc (Proprietary dependency, may require manual installation)
+ Computer Vision (For Validations)
opencv-python==4.8.0.76
+ Networking and Communication
socket (Built-in Python module)

**Compatible Operating Systems:**
The software has been tested and verified on:

+ Windows 10 / 11 (64-bit)

It may work on other systems with some modifications:

+ Ubuntu 20.04 / 22.04 LTS: Recommended for better compatibility with ML and GPU computing.
How to make it work:

Ensure Python 3.13.0 is installed (sudo apt update && sudo apt install python3.13 python3.13-venv)
Install dependencies via pip install -r requirements.txt
If using a CUDA-compatible GPU, install NVIDIA CUDA Toolkit (sudo apt install nvidia-cuda-toolkit)
Potential issue: rseriesopc (hardware communication) may not be compatible, requiring manual adaptation or an alternative OPC client.

+ macOS Monterey / Ventura (Limited compatibility due to hardware control differences)
How to make it work:

Install Python 3.13.0 using Homebrew (brew install python@3.13)
Set up a virtual environment and install dependencies (pip install -r requirements.txt)
Some 3D visualization libraries (pyglet, pyopengl, trimesh) may need manual tweaks to work with Metal instead of OpenGL.
Potential issue: rseriesopc is unlikely to function properly on macOS. Alternative OPC communication methods (such as opcua library) may be required.

**Non-Standard Hardware Usage:**
No non-standard hardware was used in this study. All computations and analyses were performed using standard desktop computing resources.

**Installation guide:**
Since Reac-Discovery consists of multiple independent scripts, there is no single executable file. Instead, you need to set up the environment and run each script as needed. Below are the steps to install dependencies and prepare your system for execution.

1. Install Python 3.13.0
Ensure you have Python 3.13.0 installed on your system. If not, install it from:

+ Windows: Python official website.
+ Ubuntu/Linux: Run:
sudo apt update && sudo apt install python3.13 python3.13-venv
+ macOS: Install via Homebrew:
brew install python@3.13

2. Verify installation:
python3 --version

2. Create a Virtual Environment (Recommended)
To avoid conflicts with other Python packages, create a virtual environment:

python3 -m venv env

Activate it:
Windows: env\Scripts\activate
Linux/macOS: source env/bin/activate

3. Install Dependencies
Inside the virtual environment, install the required dependencies:

pip install -r requirements.txt
Note: If you are using Ubuntu or macOS, and encounter OpenGL-related errors when using trimesh, install:
sudo apt install libgl1-mesa-glx  # Ubuntu
brew install mesa                 # macOS

4. Running the Scripts
Each script serves a different purpose and must be run separately. Here’s how:

+ To generate reactor geometries:
python src/Reac-Gen.py
+ To validate printability:
python src/Printability_Algorithm.py
+ To generate experimental reaction sets:
python src/Random_Reaction_Set_Algorithm.py
+ To run reaction protocols:
python src/Reaction_Protocol.py
+ To analyze NMR spectra:
python src/NMR_Spectrum_Analysis.py
+ To train ML models:
python src/ML_model_M1.py
python src/ML_model_M2.py

For more details on how to configure parameters, refer to the comments inside each script.

**Demo:**
Each script in Reac-Discovery includes instructions within its code for proper execution. The expected outputs and results are detailed in the main paper and the Supplementary Information (SI) under the experimental details section. Execution times are relatively short but vary depending on the application

**Instructions for Use - Reac-Discovery:**
Reac-Discovery is a structured methodology for designing, fabricating, and optimizing catalytic reactors in continuous flow. The process consists of several independent steps, each performed with specific scripts.

+ 1. Generating Reactor Geometries
The process begins with the generation of reactor geometries using the Reac-Gen module. In the study, nine structures were created using the gyroid geometry with different parameter values. Specifically, sizes of 10, 16, and 22 were chosen, each with three different level values to adjust thickness and porosity.

+ 2. Validating Printability and Preparing for Fabrication
Once the structures are generated, their printability can be assessed using the machine learning model implemented in ML_model_M2. A pre-trained model is provided for high-resolution photopolymerizable resin, but the Printability_Algorithm script allows users to train new models for different materials. If the reactors are validated as printable, they move on to the fabrication and functionalization stage in Reac-Fab, where they transition from CAD files to functional catalytic reactors.

+ 3. Experimental Setup in Reac-Eval
After fabrication, the reactors are placed in the Reac-Eval system, a self-driven laboratory with a multi-reactor configuration that allows continuous experimentation. To automate control of pumps, valves, reactor channels, autosamplers, and analytical techniques (such as benchtop NMR), a custom library called MASP_Library was developed.

The experimental setup begins by defining operating conditions, including temperature, concentration, gas flow, liquid flow, reactor size, and structural parameters. Using Random_Reaction_Set_Algorithm, a set of randomly selected reactions is generated for training the ML model. These reactions represent 1–3% of the total possible combinations, depending on the user's selection. The reactions are classified by reactor type and required feedstock solutions.

To execute the experiments, Reaction_Protocol uses MASP_Library to systematically conduct multiple reactions in Reac-Eval. The script reads experimental parameters from an Excel file, allowing flexibility in modifying reaction conditions.

+ 4. Reaction Analysis and Data Collection
After execution, the reaction data is analyzed using NMR_Spectrum_Analysis. This module processes NMR spectra obtained from the Magritec Spinsolve system, extracting relevant features and targets for future ML training.

+ 5. Process Optimization Using Machine Learning
Once the dataset is collected, ML_model_M1 preprocesses the data, separating and structuring it for machine learning training. This model evaluates millions of possible combinations, predicting reaction performance based on both process descriptors (e.g., flow rates, concentration, temperature) and topological descriptors (e.g., tortuosity, surface area, free volume).

The user can optimize either Space-Time Yield (STY) or Yield. The theoretical predictions are validated through Reac-Eval, comparing experimental and predicted results. If the error exceeds 5%, the model is retrained with the new experimental data. In this study, the model functioned effectively without additional adjustments, suggesting that the initial random reaction set was representative enough to model the system.

+ 6. Geometry Optimization with ML_model_M2
After optimizing process parameters, the structural descriptors are integrated into a second ML model. Features such as relative tortuosity, surface area, free volume, and packing percentage (all derived from Reac-Gen) are incorporated. This dataset is used to train ML_model_M2, which selects the best reactor geometries based on reaction performance.

Additionally, this step includes printability validation, ensuring that the proposed reactor designs can be fabricated using the selected resin and 3D printing technique. The framework can be adapted to other materials and manufacturing methods.

+ 7. Fabricating and Testing Optimized Reactors
Once the best reactor candidates are identified, Reac-Gen is used again to generate CAD files for the optimized structures. These designs are fabricated in Reac-Fab and tested in Reac-Eval to validate their real-world performance.

In the study, the top four predictions were experimentally tested in continuous flow, and their performance was compared against the ML model’s predictions. The system was further refined based on the observed accuracy of the predictions. Notably, all interactions between the ML model and experimental validation worked without additional corrections, reinforcing that the initial random reaction set was representative enough for accurate modeling.

**Additional Information**
For further details on system construction, customization, and usage, refer to the Supporting Information (SI), which includes detailed experimental procedures and additional validation steps.
