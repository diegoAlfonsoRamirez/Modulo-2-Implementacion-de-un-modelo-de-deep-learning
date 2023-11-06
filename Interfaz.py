from tensorflow.keras.models import model_from_json

# Load the model architecture from the JSON file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create the model from the loaded architecture
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import pickle

# Load the Tokenizer from the saved file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

import tkinter as tk
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create a Tkinter window
window = tk.Tk()
window.title("Complaint Dispute Prediction")
window.geometry("300x200")

# Create a label for instructions
label = tk.Label(window, text="Enter your complaint:")
label.pack()

# Create an entry field for user input
user_input = tk.Entry(window, width=50)
user_input.pack(side="top", pady=50)

# Function to make predictions
def predict_complaint_dispute():
    complaint = user_input.get()
    
    # Preprocess the user input
    max_len = 400 
    tokenizer.fit_on_texts([complaint])
    sequence = tokenizer.texts_to_sequences([complaint])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    
    # Make a prediction
    prediction = loaded_model.predict(padded_sequence)
    
    if prediction > 0.5:
        result_label.config(text="The complaint was disputed by the costumer.")
    else:
        result_label.config(text="The complaint was not disputed by the costumer.")

# Create a button to trigger predictions
predict_button = tk.Button(window, text="Predict", command=predict_complaint_dispute)
predict_button.pack()

# Create a label to display the prediction result
result_label = tk.Label(window, text="")
result_label.pack()

# Start the Tkinter main loop
window.mainloop()