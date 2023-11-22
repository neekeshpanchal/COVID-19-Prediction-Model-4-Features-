**Prototype Graphic Interface**
![image](https://github.com/neekeshpanchal/COVID-19-Prediction-Model-4-Features-/assets/80868396/f4f3ccbd-8da9-4843-86dc-25c6a9d91455)



This software implements a graphical user interface (GUI) application for training a neural network and using it to predict future COVID-19 cases. The GUI is built using the Tkinter library and includes functionalities for loading datasets, training a neural network, saving/loading trained models, and predicting future COVID-19 cases. The neural network is implemented using the scikit-learn library.

**High-Level Overview:**
- The application consists of two main classes: `NeuralNetworkGUI` for the neural network training and `CovidPredictorGUI` for predicting future COVID-19 cases.
- The neural network is trained on COVID-19 dataset features, and the trained model is saved for future use.
- The COVID-19 predictor class loads the pre-trained model and allows users to input current COVID-19 statistics to predict future cases.
- The GUI displays predictions both as text and in a bar graph, updating dynamically as predictions are made.

**Key Components:**
  - **NeuralNetworkGUI Class:**
    - Loads datasets, trains a neural network, and saves/loads trained models.
    - Displays an empty bar graph initially and updates it dynamically during predictions.

  - **CovidPredictorGUI Class:**
    - Allows users to input current COVID-19 statistics.
    - Predicts future cases using a pre-trained neural network model.
    - Displays predictions as text and updates a bar graph dynamically.

**Execution:**
- Run the script to open the GUI.
- Load a COVID-19 dataset.
- Train the neural network.
- Predict future COVID-19 cases by inputting current statistics.
- View predictions as text and in a dynamic bar graph.

dataset was pulled from this source:(https://www.kaggle.com/datasets/joebeachcapital/coronavirus-covid-19-cases-daily-updates/data)
