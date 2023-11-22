import pandas as pd
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from datetime import datetime, timedelta

class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network Trainer")

        self.scaler = None

        self.dataset_path = ""
        self.model = None

        # Interface Components
        self.label = Label(master, text="Select Dataset:")
        self.label.pack()

        self.covid_predictor = CovidPredictorGUI(master)

        self.load_button = tk.Button(master, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack()

        self.train_button = tk.Button(master, text="Train Neural Network", command=self.train_network)
        self.train_button.pack()

        self.save_button = tk.Button(master, text="Save Model", command=self.save_model)
        self.save_button.pack()

        self.load_model_button = tk.Button(master, text="Load Model", command=self.load_model_file)
        self.load_model_button.pack()

        self.prediction_label = Label(master, text="Enter Feature Values for Prediction:")
        self.prediction_label.pack()

        self.total_cases_entry = Entry(master)
        self.total_cases_entry.insert(0, "1000000")  # Initial value for total_cases
        self.total_cases_entry.pack()

        self.new_cases_entry = Entry(master)
        self.new_cases_entry.insert(0, "1000")  # Initial value for new_cases
        self.new_cases_entry.pack()

        self.total_deaths_entry = Entry(master)
        self.total_deaths_entry.insert(0, "20000")  # Initial value for total_deaths
        self.total_deaths_entry.pack()

        self.new_deaths_entry = Entry(master)
        self.new_deaths_entry.insert(0, "50")  # Initial value for new_deaths
        self.new_deaths_entry.pack()

        self.prediction_years_label = Label(master, text="Enter Years for Prediction:")
        self.prediction_years_label.pack()

        self.prediction_years_entry = Entry(master)
        self.prediction_years_entry.insert(0, "1")  # Initial value for prediction years
        self.prediction_years_entry.pack()

        self.predict_button = Button(master, text="Predict Future Cases", command=self.predict_future_cases)
        self.predict_button.pack()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack()

    def load_model_file(self):
        model_file_path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
        if model_file_path:
            self.label.config(text=f"Model Loaded: {model_file_path}")

            # Load the selected model file
            try:
                self.model = joblib.load(model_file_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading the model: {e}")
                self.model = None
                return

            # If the loaded model is trained with a scaler, use it
            if hasattr(self.model, 'scaler_'):
                self.scaler = self.model.scaler_

    def load_dataset(self):
        self.dataset_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.dataset_path:
            self.label.config(text=f"Dataset Loaded: {self.dataset_path}")

    def train_network(self):
        if not self.dataset_path:
            self.label.config(text="Please load a dataset first.")
            return

        # Load the dataset
        data = pd.read_csv(self.dataset_path)

        # Select relevant columns as features and target
        features = data[['total_cases', 'new_cases', 'total_deaths', 'new_deaths']]
        target = data['new_cases_smoothed']

        # Combine features and target for easier handling
        combined_data = pd.concat([features, target], axis=1)

        # Drop rows with missing values in both X and y
        combined_data.dropna(inplace=True)

        # Split the dataset into training and testing sets
        X_train, _, y_train, _ = train_test_split(
            combined_data[['total_cases', 'new_cases', 'total_deaths', 'new_deaths']],
            combined_data['new_cases_smoothed'],
            test_size=0.2,
            random_state=42
        )

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Create and train the neural network in a separate thread
        training_thread = threading.Thread(target=self.train_model, args=(X_train_scaled, y_train, scaler))
        training_thread.start()

    def train_model(self, X_train, y_train, scaler):
        # Create and train the neural network
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, verbose=1)
        self.model.fit(X_train, y_train)
        self.scaler = scaler

        # Plot the training loss
        self.ax.plot(self.model.loss_curve_)
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Training Loss')
        self.ax.set_title('Training Loss Over Time')

        # Update the canvas
        self.canvas.draw()

    def save_model(self):
        if self.model:
            joblib.dump(self.model, 'trained_model.joblib')
            print("Model saved successfully.")
        else:
            print("No model to save. Train the model first.")

    def load_model(self):
        try:
            loaded_model = joblib.load('trained_model.joblib')
            if isinstance(loaded_model, MLPRegressor):
                self.model = loaded_model
                print("Model loaded successfully.")
            else:
                print("Invalid model file. Please load a model trained with MLPRegressor.")
        except FileNotFoundError:
            print("No pre-trained model found. Train a new model or load a different model file.")

    def predict_future_cases(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Load a pre-trained model first.")
            return

        # Get user-input feature values
        total_cases = float(self.total_cases_entry.get())
        new_cases = float(self.new_cases_entry.get())
        total_deaths = float(self.total_deaths_entry.get())
        new_deaths = float(self.new_deaths_entry.get())

        # Standardize the input data using the available scaler
        if self.scaler:
            input_data = pd.DataFrame([[total_cases, new_cases, total_deaths, new_deaths]])
            input_data_scaled = self.scaler.transform(input_data)
        else:
            # If no scaler is available, use the model's internal scaler
            input_data = pd.DataFrame([[total_cases, new_cases, total_deaths, new_deaths]])
            input_data_scaled = self.model._transform(input_data)

        # Get the number of years for prediction
        try:
            prediction_years = float(self.prediction_years_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input for prediction years.")
            return

        # Generate future timestamps for prediction
        future_timestamps = pd.date_range(start=pd.to_datetime('today'), periods=int(prediction_years * 365), freq='D')

        # Predict future cases
        predictions = []
        for _ in range(int(prediction_years * 365)):
            prediction = self.model.predict(input_data_scaled)
            predictions.append(prediction[0])
            input_data_scaled = self.update_input_data(input_data_scaled, prediction[0])

        # Plot the predictions in real-time
        self.plot_realtime_predictions(future_timestamps, predictions)

        # Use the COVID-19 predictor to get additional predictions
        self.covid_predictor.entry_total_cases.delete(0, tk.END)
        self.covid_predictor.entry_new_cases.delete(0, tk.END)
        self.covid_predictor.entry_total_deaths.delete(0, tk.END)
        self.covid_predictor.entry_new_deaths.delete(0, tk.END)

        # Set the initial values based on the user's input
        self.covid_predictor.entry_total_cases.insert(0, str(total_cases))
        self.covid_predictor.entry_new_cases.insert(0, str(new_cases))
        self.covid_predictor.entry_total_deaths.insert(0, str(total_deaths))
        self.covid_predictor.entry_new_deaths.insert(0, str(new_deaths))

        # Run the COVID-19 predictor
        self.covid_predictor.predict()

class CovidPredictorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("COVID-19 Predictor")

        # Load the pre-trained model
        try:
            self.model = joblib.load('trained_model.joblib')
        except FileNotFoundError:
            print("Pre-trained model not found. Train a model first or provide the correct file path.")
            self.master.destroy()
            return

        # Interface Components
        self.label_total_cases = Label(master, text="Total Cases:")
        self.label_total_cases.pack()
        self.entry_total_cases = Entry(master)
        self.entry_total_cases.pack()

        self.label_new_cases = Label(master, text="New Cases:")
        self.label_new_cases.pack()
        self.entry_new_cases = Entry(master)
        self.entry_new_cases.pack()

        self.label_total_deaths = Label(master, text="Total Deaths:")
        self.label_total_deaths.pack()
        self.entry_total_deaths = Entry(master)
        self.entry_total_deaths.pack()

        self.label_new_deaths = Label(master, text="New Deaths:")
        self.label_new_deaths.pack()
        self.entry_new_deaths = Entry(master)
        self.entry_new_deaths.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.result_label = Label(master, text="")
        self.result_label.pack()

    def predict(self):
        try:
            total_cases = float(self.entry_total_cases.get())
            new_cases = float(self.entry_new_cases.get())
            total_deaths = float(self.entry_total_deaths.get())
            new_deaths = float(self.entry_new_deaths.get())
        except ValueError:
            self.result_label.config(text="Invalid input. Please enter numeric values.")
            return

        input_data = pd.DataFrame([[total_cases, new_cases, total_deaths, new_deaths]],
                                  columns=['total_cases', 'new_cases', 'total_deaths', 'new_deaths'])

        # Generate future timestamps for prediction
        future_timestamps = [
            datetime.datetime.now() + datetime.timedelta(days=1),
            datetime.datetime.now() + datetime.timedelta(weeks=1),
            datetime.datetime.now() + datetime.timedelta(weeks=4),
            datetime.datetime.now() + datetime.timedelta(weeks=26),
            datetime.datetime.now() + datetime.timedelta(weeks=52),
            datetime.datetime.now() + datetime.timedelta(weeks=104),
            datetime.datetime.now() + datetime.timedelta(weeks=156),
            datetime.datetime.now() + datetime.timedelta(weeks=208),
            datetime.datetime.now() + datetime.timedelta(weeks=260),
            datetime.datetime.now() + datetime.timedelta(weeks=312),
        ]

        predictions = []

        for timestamp in future_timestamps:
            days_difference = (timestamp - datetime.datetime.now()).days
            input_data['new_cases'] = new_cases * days_difference  # Update 'new_cases' directly
            prediction = self.model.predict(input_data)
            predictions.append((timestamp, prediction[0]))

        # Display the predictions
        result_text = ""
        for timestamp, prediction in predictions:
            result_text += f"{timestamp.strftime('%Y-%m-%d')}: Predicted New Cases (Smoothed): {prediction:.2f}\n"

        self.result_label.config(text=result_text)
    def __init__(self, master):
        self.master = master
        self.master.title("COVID-19 Predictor")

        # Load the pre-trained model
        try:
            self.model = joblib.load('trained_model.joblib')
        except FileNotFoundError:
            print("Pre-trained model not found. Train a model first or provide the correct file path.")
            self.master.destroy()
            return

        # Interface Components
        self.label_total_cases = Label(master, text="Total Cases:")
        self.label_total_cases.pack()
        self.entry_total_cases = Entry(master)
        self.entry_total_cases.pack()

        self.label_new_cases = Label(master, text="New Cases:")
        self.label_new_cases.pack()
        self.entry_new_cases = Entry(master)
        self.entry_new_cases.pack()

        self.label_total_deaths = Label(master, text="Total Deaths:")
        self.label_total_deaths.pack()
        self.entry_total_deaths = Entry(master)
        self.entry_total_deaths.pack()

        self.label_new_deaths = Label(master, text="New Deaths:")
        self.label_new_deaths.pack()
        self.entry_new_deaths = Entry(master)
        self.entry_new_deaths.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.result_label = Label(master, text="")
        self.result_label.pack()

    def predict(self):
        try:
            total_cases = float(self.entry_total_cases.get())
            new_cases = float(self.entry_new_cases.get())
            total_deaths = float(self.entry_total_deaths.get())
            new_deaths = float(self.entry_new_deaths.get())
        except ValueError:
            self.result_label.config(text="Invalid input. Please enter numeric values.")
            return

        input_data = pd.DataFrame([[total_cases, new_cases, total_deaths, new_deaths]],
                                  columns=['total_cases', 'new_cases', 'total_deaths', 'new_deaths'])

        # Generate future timestamps for prediction
        future_timestamps = [
            datetime.now() + timedelta(days=1),
            datetime.now() + timedelta(weeks=1),
            datetime.now() + timedelta(weeks=4),
            datetime.now() + timedelta(weeks=26),
            datetime.now() + timedelta(weeks=52),
            datetime.now() + timedelta(weeks=104),
            datetime.now() + timedelta(weeks=156),
            datetime.now() + timedelta(weeks=208),
            datetime.now() + timedelta(weeks=260),
            datetime.now() + timedelta(weeks=312),
        ]

        predictions = []

        for timestamp in future_timestamps:
            days_difference = (timestamp - datetime.now()).days
            input_data['new_cases'] = new_cases * days_difference  # Update 'new_cases' directly
            prediction = self.model.predict(input_data)
            predictions.append((timestamp, prediction[0]))

        # Display the predictions
        result_text = ""
        for timestamp, prediction in predictions:
            result_text += f"{timestamp.strftime('%Y-%m-%d')}: Predicted New Cases (Smoothed): {prediction:.2f}\n"

        self.result_label.config(text=result_text)
        try:
            total_cases = float(self.entry_total_cases.get())
            new_cases = float(self.entry_new_cases.get())
            total_deaths = float(self.entry_total_deaths.get())
            new_deaths = float(self.entry_new_deaths.get())
        except ValueError:
            self.result_label.config(text="Invalid input. Please enter numeric values.")
            return

        input_data = pd.DataFrame([[total_cases, new_cases, total_deaths, new_deaths]])
        prediction = self.model.predict(input_data)

        self.result_label.config(text=f"Predicted New Cases (Smoothed): {prediction[0]:.2f}")


# Create the main application window
root = tk.Tk()
app = NeuralNetworkGUI(root)

# Start the GUI application
root.mainloop()
