import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa

# Ορίζουμε μια λίστα που περιέχει τα ονόματα-ετικέτες των φακέλων που βρίσκονται στο dataset
# και χρησιμοποιούνται για να συλλέξουμε όλα τα δεδομένα.
datasetDirFolder = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Ορίζουμε το γενικό directory όπου βρίσκονται όλα τα δεδομένα μας.
datasetDirectory = 'Data/SpeechCommands/speech_commands_v0.02'

# Καθορίζουμε τον αριθμό των δειγμάτων ήχου που θα χρησιμοποιήσουμε για επεξεργασία.
samples = 2000

# Φορτώνουμε τα δεδομένα ήχου.
def load_data(datasetDirFolder, datasetDirectory, samples):
    audio_fullpaths = []
    audio_digit_labels = []

    # Επανάληψη μέσω των φακέλων στο dataset.
    for digitFolder in datasetDirFolder:
        # Δημιουργία του απόλυτου μονοπατιού για κάθε φάκελο (ψηφίου).
        digit_fullpath = os.path.join(datasetDirectory, digitFolder)
        # Λίστα με τα αρχεία ήχου σε κάθε φάκελο (ψηφίου).
        digit_file_names = os.listdir(digit_fullpath)
        # Πλήρες μονοπάτι για κάθε αρχείο ήχου σε αυτό το φάκελο.
        files_fullpath = [os.path.join(digit_fullpath, file) for file in digit_file_names]

        audio_fullpaths.extend(files_fullpath)
        # Επισημείωση των αρχείων ήχου με το αντίστοιχο ψηφίο (ετικέτα).
        audio_digit_labels.extend([digitFolder] * len(files_fullpath))

    # Τυχαία επιλογή δειγμάτων για επεξεργασία.
    random.seed(42)
    random_selected_samples = random.sample(range(len(audio_fullpaths)), samples)

    selected_audio_fullpaths = []
    selected_audio_digit_labels = []

    for x in random_selected_samples:
        selected_audio_fullpaths.append(audio_fullpaths[x])
        selected_audio_digit_labels.append(audio_digit_labels[x])

    print("Total audio samples:", len(selected_audio_fullpaths))

    return selected_audio_fullpaths, selected_audio_digit_labels

# Εξαγωγή των χαρακτηριστικών MFCC από τα αρχεία ήχου.
def extract_mfcc_features(audio_paths):
    mfcc_features = []
    max_mfcc_length = 0

    for i, path in enumerate(audio_paths):
        print(f"Processing audio {i + 1}/{len(audio_paths)}")

        # Φόρτωση του ήχου και προεπεξεργασία.
        audio_signal, _ = librosa.load(path, sr=16000)
        preemphasized_audio = librosa.effects.preemphasis(audio_signal)
        filtered_audio, _ = librosa.effects.trim(preemphasized_audio, top_db=20)

        # Υπολογισμός των χαρακτηριστικών MFCC.
        mfcc = librosa.feature.mfcc(filtered_audio, sr=16000, n_mfcc=13)
        mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        mfcc_features.append(mfcc_normalized)

        # Εύρεση του μέγιστου μήκους των MFCC χαρακτηριστικών.
        current_mfcc_length = mfcc_normalized.shape[1]
        if current_mfcc_length > max_mfcc_length:
            max_mfcc_length = current_mfcc_length

    print("Feature extraction completed.")
    return mfcc_features, max_mfcc_length

# Φόρτωση και επεξεργασία των δεδομένων ήχου.
selected_audio_fullpaths, selected_audio_digit_labels = load_data(datasetDirFolder, datasetDirectory, samples)
mfcc_features, max_mfcc_length = extract_mfcc_features(selected_audio_fullpaths)

# Προσαρμογή των ακολουθιών MFCC χαρακτηριστικών.
X = []
for mfcc in mfcc_features:
    remained_length = max_mfcc_length - mfcc.shape[1]
    padded_mfcc = np.pad(mfcc, ((0, 0), (0, remained_length)), mode='constant', constant_values=0)
    X.append(padded_mfcc.T)
X = np.array(X)

# Φόρτωση του αρχείου recording.wav και εξαγωγή των χαρακτηριστικών MFCC.
recording_path = "recording.wav"
audio_signal, _ = librosa.load(recording_path, sr=16000)
preemphasized_audio = librosa.effects.preemphasis(audio_signal)
filtered_audio, _ = librosa.effects.trim(preemphasized_audio, top_db=20)
mfcc = librosa.feature.mfcc(filtered_audio, sr=16000, n_mfcc=13)
mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)
padded_mfcc = np.pad(mfcc_normalized, ((0, 0), (0, max_mfcc_length - mfcc_normalized.shape[1])), mode='constant', constant_values=0)
X_recording = np.array([padded_mfcc.T])

# Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και ελέγχου.
# Υποθέτουμε ότι έχουμε ήδη τους πίνακες X_train, X_test, y_train, y_test από προηγούμενη χρήση της train_test_split.

# Δημιουργία και εκπαίδευση ενός μοντέλου νευρωνικού δικτύου.
model = models.Sequential([
    layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Αξιολόγηση της ακρίβειας του μοντέλου στα δεδομένα ελέγχου.
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Πρόβλεψη του ψηφίου για το αρχείο recording.wav
predicted_label = np.argmax(model.predict(X_recording))
print('Predicted digit for recording.wav:', predicted_label)
