# natHacks2025 - Neuromotion
An application that uses the OpenBCI headset to record EEG signals that capture motor imagery and feeds them into a machine learning model to predict intended movements. 

## Built With
* [![Python][Python.org]][Python-url]
* [![HTML5][HTML5.badge]][HTML5-url]

[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[HTML5.badge]: https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white
[HTML5-url]: https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5

## Running The Application
1. Put on OpenBCI headset and secure it tightly onto head.
2. Run GUI/app.py
3. Open the browser and access localhost:5000.

## Python Packages
   ### [MNE](https://mne.tools/stable/index.html)
   ```py
   pip install mne
   ```
   > A powerful library for processing, analyzing, and visualizing EEG data. Used to filter raw brain signals, extract relevant epochs, and prepare data for machine learning classification.
   ### [Pytorch](https://pytorch.org/)
   ```py
   pip install pytorch
   ```
   > Used to build and train the machine learning model that classifies EEG motor imagery signals. 
   ### [Brainflow](https://brainflow.org/get_started/?platform=windows&language=python&environ=pip&) 
   ```py
   pip install brainflow
   ```
   > Handles communication with the OpenBCI headset and streams EEG data for processing. 
   ### [Eventlet](https://eventlet.readthedocs.io/en/latest/) 
   ```py
   pip install eventlet
   ```
   > Enables real-time data streaming and asynchronous communication between the backend and GUI. 
   ### [Numpy](https://numpy.org/) 
   ```py
   pip install numpy
   ```
   > Provides fast numerical operations and efficient handling of EEG data arrays. 
   ### [Threading](https://docs.python.org/3/library/threading.html)
   ```py
   pip install threading
   ```
   > A built-in Python module used to run background tasks (like real-time EEG data collection) alongside the main program without freezing the interface.
   ### [Time](https://docs.python.org/3/library/time.html)
   ```py
   pip install time
   ```
   > A built-in Python module used for timing operations, synchronization, and delays during data acquisition and processing.
   ### [Flask](https://flask.palletsprojects.com/en/stable)
   ```py
   pip install flask
   ```
   > Powers the web-based interface, connecting the machine learning model to the user-facing GUI. 
   ### [Flask_SocketIO](https://flask-socketio.readthedocs.io/en/latest/)
   ```py
   pip install flask_socketio
   ```
   > Enables real-time communication between the EEG processing backend and the GUI, allowing live updates of predicted movements. 
   ### [sklearn](https://scikit-learn.org/stable/)
   ```py
   pip install sklearn
   ```
   > Used for preprocessing EEG data, extracting features, and implementing classification algorithms such as Linear Discriminant Analysis (LDA) to predict motor imagery movements.

## Materials
[Presentation slides](https://docs.google.com/presentation/d/1h2WjvpodQrcb_n5vvOtkd0tqTo_iaAlo-Vs2vEI9-z0/edit?usp=sharing) - Slides used for presentation. \
[Training dataset](https://physionet.org/content/eegmmidb/1.0.0/)  - The data we used to train the machine learning model.

## License
Distributed under the project_license. See `license.txt` for more information.

