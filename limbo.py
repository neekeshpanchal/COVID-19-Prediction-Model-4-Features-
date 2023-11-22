import tkinter as tk
from tkinter import Entry, Label, Button
import joblib
import pandas as pd

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
        self.label_total_cases.grid(row=0, column=0)
        self.entry_total_cases = Entry(master)
        self.entry_total_cases.grid(row=0, column=1)

        self.label_new_cases = Label(master, text="New Cases:")
        self.label_new_cases.grid(row=1, column=0)
        self.entry_new_cases = Entry(master)
        self.entry_new_cases.grid(row=1, column=1)

        self.label_total_deaths = Label(master, text="Total Deaths:")
        self.label_total_deaths.grid(row=2, column=0)
        self.entry_total_deaths = Entry(master)
        self.entry_total_deaths.grid(row=2, column=1)

        self.label_new_deaths = Label(master, text="New Deaths:")
        self.label_new_deaths.grid(row=3, column=0)
        self.entry_new_deaths = Entry(master)
        self.entry_new_deaths.grid(row=3, column=1)

        self.predict_button = Button(master, text="Predict", command=self.predict)
        self.predict_button.grid(row=4, column=0, columnspan=2)

        self.result_label = Label(master, text="")
        self.result_label.grid(row=5, column=0, columnspan=2)

    def predict(self):
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


if __name__ == "__main__":
    root = tk.Tk()
    app = CovidPredictorGUI(root)
    root.mainloop()
