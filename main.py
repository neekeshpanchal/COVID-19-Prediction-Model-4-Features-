import pandas as pd
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, messagebox, Frame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import matplotlib.ticker as ticker

class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network Trainer")

        self.graph_frame = Frame(master, bg="black")
        self.graph_frame.pack(expand=True, fill="both")

        self.bar_graph_frame = Frame(master, bg="black")
        self.bar_graph_frame.pack(expand=True, fill="both")

        # Create and display an empty bar graph initially
        self.plot_empty_bar_graph()

        # Set the initial size of the window
        self.master.geometry("800x600")

        self.master.configure(bg="black")  # Set background color

        self.scaler = None

        self.dataset_path = ""
        self.model = None

        # Interface Components
        self.label = Label(master, text="Select Dataset:", bg="black", fg="green")
        self.label.pack()

        self.covid_predictor = CovidPredictorGUI(master, self)

        self.load_button = tk.Button(master, text="Load Dataset", command=self.load_dataset, bg="green", fg="black")
        self.load_button.pack()

        self.train_button = tk.Button(master, text="Train Neural Network", command=self.train_network, bg="green", fg="black")
        self.train_button.pack()

        self.save_button = tk.Button(master, text="Save Model", command=self.save_model, bg="green", fg="black")
        self.save_button.pack()

        self.load_model_button = tk.Button(master, text="Load Model", command=self.load_model_file, bg="green", fg="black")
        self.load_model_button.pack()

        self.graph_frame = Frame(master, bg="black")
        self.graph_frame.pack(expand=True, fill="both")

        self.bar_graph_frame = Frame(master, bg="black")
        self.bar_graph_frame.pack(expand=True, fill="both")

    def plot_empty_bar_graph(self):
        fig, ax = plt.subplots()

    # Bar graph with black background and no bars (empty)
        ax.bar([], [], color='green')
        ax.set_facecolor('black')  # Set background color
        ax.set_xlabel('Time for Prediction', color='green')
        ax.set_ylabel('Predicted New Cases (Smoothed)', color='green')
        ax.set_title('Predictions Over Time', color='green')

    # Embed the empty bar graph in the GUI
        self.bar_canvas = FigureCanvasTkAgg(fig, master=self.bar_graph_frame)
        self.bar_canvas.get_tk_widget().pack(expand=True, fill="both")

    # Update the canvas
        self.bar_canvas.draw()

    # Store the axes for later use in updating the bar graph
        self.bar_axes = ax

    def update_bar_graph(self, predictions):
        # Clear previous bars
        self.bar_axes.clear()

        # Bar graph with black background and green bars
        bars = self.bar_axes.bar(predictions.keys(), predictions.values(), color='green')

        # Set background color
        self.bar_axes.set_facecolor('black')

        # Set labels and title
        self.bar_axes.set_xlabel('Time for Prediction', color='green')
        self.bar_axes.set_ylabel('Predicted New Cases (Smoothed) (in units)', color='green')  # Update y-axis label
        self.bar_axes.set_title('Predictions Over Time', color='green')

        # Set y-axis scale according to the values
        self.bar_axes.set_ylim(min(predictions.values()), max(predictions.values()) + max(predictions.values()) * 0.1)

        # Format y-axis labels based on the magnitude of values
        self.bar_axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

        # Add tooltips on hover
        def on_hover(event):
            if event.inaxes == self.bar_axes:
                for bar, (key, value) in zip(bars, predictions.items()):
                    cont, ind = bar.contains(event)
                    if cont:
                        tooltip_text = f"{key}: {int(value):,} units"
                        self.tooltip.set_text(tooltip_text)
                        self.tooltip.set_visible(True)
                        self.tooltip.set_position((event.xdata, event.ydata))
                        self.bar_canvas.draw_idle()
                        break
                else:
                    self.tooltip.set_visible(False)
                    self.bar_canvas.draw_idle()

        # Attach the hover event
        self.bar_canvas.mpl_connect("motion_notify_event", on_hover)

        # Embed the bar graph in the GUI
        self.bar_canvas.draw()
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

        # Plot the training loss (optional)
        self.plot_training_loss()

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


class CovidPredictorGUI:
    def __init__(self, master, main_gui):
        self.master = master
        self.main_gui = main_gui
        self.master.title("COVID-19 Predictor")

        # Load the pre-trained model
        try:
            self.model = joblib.load('trained_model.joblib')
        except FileNotFoundError:
            print("Pre-trained model not found. Train a model first or provide the correct file path.")
            return

        # Interface Components
        self.label_total_cases = Label(master, text="Total Cases:", bg="black", fg="green")
        self.label_total_cases.pack()
        self.entry_total_cases = Entry(master)
        self.entry_total_cases.pack()

        self.label_new_cases = Label(master, text="New Cases:", bg="black", fg="green")
        self.label_new_cases.pack()
        self.entry_new_cases = Entry(master)
        self.entry_new_cases.pack()

        self.label_total_deaths = Label(master, text="Total Deaths:", bg="black", fg="green")
        self.label_total_deaths.pack()
        self.entry_total_deaths = Entry(master)
        self.entry_total_deaths.pack()

        self.label_new_deaths = Label(master, text="New Deaths:", bg="black", fg="green")
        self.label_new_deaths.pack()
        self.entry_new_deaths = Entry(master)
        self.entry_new_deaths.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict, bg="green", fg="black")
        self.predict_button.pack()

        self.result_label = Label(master, text="", bg="black", fg="green")
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

        predictions = {}

        for timestamp in future_timestamps:
            days_difference = (timestamp - datetime.now()).days
            input_data['new_cases'] = new_cases * days_difference  # Update 'new_cases' directly
            prediction = self.model.predict(input_data)
            predictions[timestamp.strftime('%Y-%m-%d')] = prediction[0]

        # Display the predictions as text
        result_text = ""
        for timestamp, prediction in predictions.items():
            result_text += f"{timestamp}: Predicted New Cases (Smoothed): {prediction:.2f}\n"

        self.result_label.config(text=result_text)
        self.main_gui.update_bar_graph(predictions)

# Create the main application window
root = tk.Tk()
app = NeuralNetworkGUI(root)

# Start the GUI application
root.mainloop()
