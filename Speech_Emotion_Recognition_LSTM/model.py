import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import librosa

# Load the saved model
model = load_model('trymodel.h5')

# Function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Assuming you have a test dataset in the same directory format as the training dataset
test_paths = []
test_labels = []
for dirname, _, filenames in os.walk('/kaggle/input/toronto-emotional-speech-set-tess'):
    for filename in filenames:
        test_paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        test_labels.append(label.lower())

# Create a dataframe for the test dataset
test_df = pd.DataFrame()
test_df['speech'] = test_paths
test_df['label'] = test_labels

# Extract MFCC features for the test dataset
X_test_mfcc = test_df['speech'].apply(lambda x: extract_mfcc(x))

X_test = [x for x in X_test_mfcc]
X_test = np.array(X_test)
X_test = np.expand_dims(X_test, -1)

# One-hot encode the labels for the test dataset
enc = OneHotEncoder()
y_test = enc.fit_transform(test_df[['label']])
y_test = y_test.toarray()

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
