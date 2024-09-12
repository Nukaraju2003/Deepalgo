import json
import numpy as np
from keras.models import model_from_json

# Load the model architecture from the JSON file
with open("jsonfiles\\model_architecture.json", "r") as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)

# Load the model weights from the JSON file
with open("jsonfiles\\model_weights.json", "r") as json_file:
    weights_serializable = json.load(json_file)

# Convert the weights back to numpy arrays
model_weights = [np.array(w) for w in weights_serializable]

# Set the weights in the loaded model
loaded_model.set_weights(model_weights)

# Compile the model (use the same settings as before)
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the complete model to an .h5 file
loaded_model.save("h5 models\\work\\model.h5")

print("Model loaded, compiled, and saved to model.h5")
