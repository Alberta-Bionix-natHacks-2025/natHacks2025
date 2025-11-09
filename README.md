# natHacks2025 - Neuromotion
Alberta Bionix's natHacks 2025 project repository

## Running The Application
1. Put on OpenBCI headset and secure it tightly onto head
2. Run app.py
3. Open the browser and access localhost:5000

Include a ReadMe file
explaining your project,
how to access it, and any
extra materials (slides,
links, credits).

## Python Packages
   ### [Pytorch](https://pytorch.org/)
   > Used to build and train the machine learning model that classifies EEG motor imagery signals.
   ### [Brainflow](https://brainflow.org/get_started/?platform=windows&language=python&environ=pip&) 
   > Handles communication with the OpenBCI headset and streams EEG data for processing.
   ### [Eventlet](https://eventlet.readthedocs.io/en/latest/) 
   > Enables real-time data streaming and asynchronous communication between the backend and GUI.
   ### [Numpy](https://numpy.org/) 
   > Provides fast numerical operations and efficient handling of EEG data arrays.
   ### [Flask](https://flask.palletsprojects.com/en/stable)
   > Powers the web-based interface, connecting the machine learning model to the user-facing GUI.

## Materials
[Presentation slides](https://docs.google.com/presentation/d/1h2WjvpodQrcb_n5vvOtkd0tqTo_iaAlo-Vs2vEI9-z0/edit?usp=sharing) \
[Training dataset](https://physionet.org/content/eegmmidb/1.0.0/) 

## Credits
