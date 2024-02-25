import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa

# define a list containing the names-labels of the folders located in the dataset
# and are used to collect all the data
datasetDirFolder = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# define the general directory where all our data is located
datasetDirectory = 'Data/SpeechCommands/speech_commands_v0.02'

# determine the number of sound samples that will be used for processing
samples = 2000

# load the audio data
def load_data(datasetDirFolder, datasetDirectory, samples):
    audio_fullpaths = []
    audio_digit_labels = []

    # loop through the folders in the dataset
    for digitFolder in datasetDirFolder:
        # create the absolute path for each folder (digit)
        digit_fullpath = os.path.join(datasetDirectory, digitFolder)
        # list of audio files in each folder (digit)
        digit_file_names = os.listdir(digit_fullpath)
        # full path for each audio file in this folder
        files_fullpath = [os.path.join(digit_fullpath, file) for file in digit_file_names]

        audio_fullpaths.extend(files_fullpath)
        # labeling the audio files with the corresponding digit (label)
        audio_digit_labels.extend([digitFolder] * len(files_fullpath))

    # random selection of samples for processing
    random.seed(42)
    random_selected_samples = random.sample(range(len(audio_fullpaths)), samples)

    selected_audio_fullpaths = []
    selected_audio_digit_labels = []

    for x in random_selected_samples:
        selected_audio_fullpaths.append(audio_fullpaths[x])
        selected_audio_digit_labels.append(audio_digit_labels[x])

    print("Total audio samples:", len(selected_audio_fullpaths))

    return selected_audio_fullpaths, selected_audio_digit_labels

# extraction of MFCC features from the audio files
def extract_mfcc_features(audio_paths):
    mfcc_features = []
    max_mfcc_length = 0

    for i, path in enumerate(audio_paths):
        print(f"Processing audio {i + 1}/{len(audio_paths)}")

        # load the audio and preprocessing
        audio_signal, _ = librosa.load(path, sr=16000)
        preemphasized_audio = librosa.effects.preemphasis(audio_signal)
        filtered_audio, _ = librosa.effects.trim(preemphasized_audio, top_db=20)

        # calculation of MFCC features
        mfcc = librosa.feature.mfcc(filtered_audio, sr=16000, n_mfcc=13)
        mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        mfcc_features.append(mfcc_normalized)

        # finding the maximum length of MFCC features
        current_mfcc_length = mfcc_normalized.shape[1]
        if current_mfcc_length > max_mfcc_length:
            max_mfcc_length = current_mfcc_length

    print("Feature extraction completed")
    return mfcc_features, max_mfcc_length

# load and process the audio data
selected_audio_fullpaths, selected_audio_digit_labels = load_data(datasetDirFolder, datasetDirectory, samples)
mfcc_features, max_mfcc_length = extract_mfcc_features(selected_audio_fullpaths)

# adjusting the MFCC feature sequences
X = []
for mfcc in mfcc_features:
    remained_length = max_mfcc_length - mfcc.shape[1]
    padded_mfcc = np.pad(mfcc, ((0, 0), (0, remained_length)), mode='constant', constant_values=0)
    X.append(padded_mfcc.T)
X = np.array(X)

# load the recording.wav file and extract the MFCC features
recording_path = "recording.wav"
audio_signal, _ = librosa.load(recording_path, sr=16000)
preemphasized_audio = librosa.effects.preemphasis(audio_signal)
filtered_audio, _ = librosa.effects.trim(preemphasized_audio, top_db=20)
mfcc = librosa.feature.mfcc(filtered_audio, sr=16000, n_mfcc=13)
mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)
padded_mfcc = np.pad(mfcc_normalized, ((0, 0), (0, max_mfcc_length - mfcc_normalized.shape[1])), mode='constant', constant_values=0)
X_recording = np.array([padded_mfcc.T])

# splitting the data into training and testing sets
# we assume that we already have the arrays X_train, X_test, y_train, y_test from a previous use of the train_test_split

# creating and training a neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# evaluation of the model's accuracy on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# prediction of the digit for the recording.wav file
predicted_label = np.argmax(model.predict(X_recording))
print('Predicted digit for recording.wav:', predicted_label)
