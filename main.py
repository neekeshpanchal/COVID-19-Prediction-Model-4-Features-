import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network Trainer")

        self.dataset_path = ""
        self.model = None

        # Interface Components
        self.label = tk.Label(master, text="Select Dataset:")
        self.label.pack()

        self.load_button = tk.Button(master, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack()

        self.train_button = tk.Button(master, text="Train Neural Network", command=self.train_network)
        self.train_button.pack()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack()

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
        training_thread = threading.Thread(target=self.train_model, args=(X_train_scaled, y_train))
        training_thread.start()

    def train_model(self, X_train, y_train):
        # Create and train the neural network
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, verbose=1)
        self.model.fit(X_train, y_train)

        # Plot the training loss
        self.ax.plot(self.model.loss_curve_)
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Training Loss')
        self.ax.set_title('Training Loss Over Time')

        # Update the canvas
        self.canvas.draw()

# Create the main application window
root = tk.Tk()
app = NeuralNetworkGUI(root)

# Start the GUI application
root.mainloop()
