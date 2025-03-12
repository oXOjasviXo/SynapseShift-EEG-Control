import os
import time
import threading
import pickle
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pynput.keyboard import Controller, Key
import tkinter as tk

# Import BrainFlow modules for real EEG acquisition
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# For deep learning option – using TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# ----------------------------
# GPU Setup for TensorFlow
# ----------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth on each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

# ----------------------------
# Global Settings and Variables
# ----------------------------
SIMULATED = False         # Use simulated data if True or if board initialization fails
USE_DEEP_LEARNING = False # Toggle between SVM and deep learning classifier
running = False           # Flag to control real-time EEG processing loop

# Map actions to keyboard keys (for "none", no key is pressed)
ACTIONS = {
    "none": None,
    "blink": Key.enter,
    "jaw": Key.right,
    "relax": Key.space,
    "motor": Key.left,
    "mental": Key.shift
}

# Mapping between action names and numeric labels
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS.keys())}
INDEX_TO_ACTION = {idx: action for action, idx in ACTION_TO_INDEX.items()}

# Global variables for the classifier and its statistics
classifier = None
model_stats = {}

# Global variables for BrainFlow board access
board = None
BOARD_ID = None

# ----------------------------
# Simulated EEG Data Generator
# ----------------------------
def generate_simulated_eeg(n_channels, n_samples, fs=250):
    """
    Generates simulated EEG data by summing sine waves in the alpha (10 Hz),
    beta (20 Hz), and gamma (40 Hz) ranges with added noise.
    """
    t = np.arange(n_samples) / fs
    eeg_data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        alpha = 10 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        beta  = 5 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        gamma = 2 * np.sin(2 * np.pi * 40 * t + np.random.rand() * 2 * np.pi)
        noise = np.random.randn(n_samples) * 0.5
        eeg_data[ch, :] = alpha + beta + gamma + noise
    return eeg_data

# ----------------------------
# BrainFlow Device Initialization and Cleanup
# ----------------------------
def initialize_board():
    """
    Attempts to initialize the BrainFlow session for the OpenBCI Cyton.
    Falls back to simulated data if an error occurs.
    """
    global board, BOARD_ID, SIMULATED
    try:
        params = BrainFlowInputParams()
        # Uncomment and adjust if a specific serial port is needed:
        # params.serial_port = '/dev/ttyUSB0'  # Linux or 'COM3' for Windows
        BOARD_ID = BoardIds.CYTON_BOARD.value
        board = BoardShim(BOARD_ID, params)
        BoardShim.enable_dev_board_logger()
        board.prepare_session()
        board.start_stream()
        print("BrainFlow session started on OpenBCI Cyton.")
    except Exception as e:
        print(f"Error initializing board: {e}\nFalling back to simulated EEG data.")
        SIMULATED = True
        board = None

def close_board():
    """
    Stops the BrainFlow data stream and releases the session.
    """
    global board
    if board is not None:
        board.stop_stream()
        board.release_session()
        board = None
        print("BrainFlow session closed.")

# ----------------------------
# EEG Data Acquisition and Processing
# ----------------------------
def get_filtered_eeg():
    """
    Acquires a 0.5-second (125 samples at 250 Hz) window of EEG data (via BrainFlow or simulated)
    and applies a 5th-order Butterworth bandpass filter (1-50 Hz).
    """
    fs = 250
    n_samples = int(fs * 0.5)
    if not SIMULATED and board is not None:
        try:
            data = board.get_current_board_data(n_samples)
            eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
            eeg_data = data[eeg_channels, :]
            print("Acquired EEG data from OpenBCI Cyton.")
        except Exception as e:
            print(f"Error acquiring data from board: {e}\nUsing simulated data instead.")
            eeg_data = generate_simulated_eeg(8, n_samples, fs)
    else:
        eeg_data = generate_simulated_eeg(8, n_samples, fs)

    b, a = signal.butter(5, [1.0 / (0.5 * fs), 50.0 / (0.5 * fs)], btype='band')
    filtered_data = signal.filtfilt(b, a, eeg_data, axis=1)
    return filtered_data

def extract_features(eeg_data):
    """
    Computes features by calculating the average power in the alpha (8-12 Hz),
    beta (12-30 Hz), and gamma (30-50 Hz) bands for each channel.
    Returns a feature vector.
    """
    fs = 250
    features = []
    for channel in eeg_data:
        f, Pxx = signal.welch(channel, fs=fs, nperseg=len(channel))
        alpha_power = np.mean(Pxx[(f >= 8) & (f < 12)])
        beta_power  = np.mean(Pxx[(f >= 12) & (f < 30)])
        gamma_power = np.mean(Pxx[(f >= 30) & (f <= 50)])
        features.extend([alpha_power, beta_power, gamma_power])
    return np.array(features)

# ----------------------------
# Model Saving and Loading Utility
# ----------------------------
def save_model(model):
    """
    Saves the trained model to the 'models/' directory.
    For deep learning models, saves as an HDF5 file; for SVM models, uses pickle.
    """
    if USE_DEEP_LEARNING:
        model.save("models/model_dl.h5")
        print("Deep learning model saved to models/model_dl.h5")
    else:
        with open("models/model_svm.pkl", "wb") as f:
            pickle.dump(model, f)
        print("SVM model saved to models/model_svm.pkl")

def load_model():
    """
    Attempts to load an existing model from the 'models/' directory.
    Loads the deep learning model if USE_DEEP_LEARNING is True; otherwise loads the SVM model.
    """
    global classifier
    if USE_DEEP_LEARNING:
        model_path = "models/model_dl.h5"
        if os.path.exists(model_path):
            classifier = tf.keras.models.load_model(model_path)
            print("Loaded deep learning model from", model_path)
        else:
            print("No deep learning model found.")
    else:
        model_path = "models/model_svm.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                classifier = pickle.load(f)
            print("Loaded SVM model from", model_path)
        else:
            print("No SVM model found.")

# ----------------------------
# Classification and Action Execution
# ----------------------------
def classify_action(model, features):
    """
    Uses the given model to predict the action from the feature vector.
    Returns the corresponding action label.
    """
    features = features.reshape(1, -1)
    if USE_DEEP_LEARNING:
        pred_prob = model.predict(features)
        pred_index = np.argmax(pred_prob, axis=1)[0]
    else:
        pred_index = model.predict(features)[0]
    return INDEX_TO_ACTION.get(pred_index, None)

def perform_action(action):
    """
    Simulates a keyboard press for the specified action.
    If the action is "none", no key is pressed.
    """
    if action == "none":
        print("No action performed.")
        return
    keyboard = Controller()
    key_to_press = ACTIONS[action]
    keyboard.press(key_to_press)
    keyboard.release(key_to_press)
    print(f"Performed action: {action}")

# ----------------------------
# Model Training and Evaluation
# ----------------------------
def train_model():
    """
    Trains a classifier (SVM or deep learning) using collected data.
    Computes performance metrics, saves the model, and prints statistics.
    """
    global classifier, model_stats
    data_file = "data/calibration_data.pkl"
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        X = np.array(data["features"])
        y = np.array(data["labels"])
        if USE_DEEP_LEARNING:
            num_classes = len(ACTIONS)
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X.shape[1],)),
                Dense(32, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y, epochs=20, verbose=0)
        else:
            model = SVC(probability=True, decision_function_shape='ovr')
            model.fit(X, y)
        classifier = model
        save_model(model)
        if USE_DEEP_LEARNING:
            y_pred = np.argmax(model.predict(X), axis=1)
        else:
            y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        model_stats = {"accuracy": acc, "precision": prec, "recall": rec}
        print(f"Trained classifier: {model_stats}")
    else:
        print("No collection data found. Please collect data first.")

def show_model_stats():
    """
    Updates the model statistics display variable with current performance metrics.
    """
    if model_stats:
        stats_str = (f"Accuracy: {model_stats['accuracy']:.2f}\n"
                     f"Precision: {model_stats['precision']:.2f}\n"
                     f"Recall: {model_stats['recall']:.2f}")
    else:
        stats_str = "No model stats available."
    model_stats_var.set(stats_str)

# ----------------------------
# Real-Time EEG Processing and Visualization
# ----------------------------
def process_eeg_commands():
    """
    Continuously collects EEG data every 0.5 seconds, extracts features,
    uses the classifier to predict the action, and performs the corresponding keyboard action.
    """
    global running, classifier
    while running:
        eeg_data = get_filtered_eeg()
        features = extract_features(eeg_data)
        if classifier is not None:
            predicted_action = classify_action(classifier, features)
            if predicted_action:
                perform_action(predicted_action)
        time.sleep(0.5)

def visualize_eeg():
    """
    Displays a real-time plot of the last 10 seconds of EEG data for all 8 channels.
    A rolling buffer of 10 seconds (2,500 samples at 250 Hz) is maintained for each channel.
    Each channel is plotted in its own subplot.
    The x-axis shows time from -10 to 0 seconds (with 0 being the current time),
    and the y-axis is set to [-20, 20].
    """
    fs = 250
    window_duration = 10  # seconds
    n_total = int(fs * window_duration)  # total samples for 10 seconds (2500 samples)
    # Initialize a buffer for each of the 8 channels
    buffer = np.zeros((8, n_total))
    time_axis = np.linspace(-window_duration, 0, n_total)
    
    plt.ion()
    fig, axs = plt.subplots(8, 1, sharex=True, figsize=(10, 12))
    lines = []
    for i in range(8):
        line, = axs[i].plot(time_axis, buffer[i])
        axs[i].set_ylim(-40, 40)
        axs[i].set_ylabel(f"Ch {i+1}")
        lines.append(line)
    axs[-1].set_xlabel("Time (sec)")
    
    while running:
        new_data = get_filtered_eeg()  # new_data shape: (8, 125)
        # Update the buffer for each channel
        for i in range(8):
            buffer[i] = np.roll(buffer[i], -new_data.shape[1])
            buffer[i, -new_data.shape[1]:] = new_data[i]
            lines[i].set_ydata(buffer[i])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.5)
    plt.close(fig)

# ----------------------------
# Data Collection (Previously Calibration)
# ----------------------------
def start_collection():
    """
    Collects EEG data for each defined action (including "none").
    For each action, the user is prompted to get ready, then 10 samples are collected
    (with a 1-second interval between samples). After collection, the user chooses whether
    to include the collected data in the dataset.
    """
    data_file = "data/calibration_data.pkl"
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = {"features": [], "labels": []}
    
    for action in ACTIONS.keys():
        tk.messagebox.showinfo("Data Collection", f"Get ready to perform '{action}'. Click OK when ready.")
        print(f"Collecting data for {action}...")
        temp_features = []
        temp_labels = []
        for i in range(10):
            print(f"Collecting sample {i+1} of 10 for {action}...")
            eeg_data = get_filtered_eeg()
            features = extract_features(eeg_data)
            temp_features.append(features)
            temp_labels.append(ACTION_TO_INDEX[action])
            time.sleep(1.0)
        include = tk.messagebox.askyesno("Include Data", f"Include data for '{action}' in dataset?")
        if include:
            data["features"].extend(temp_features)
            data["labels"].extend(temp_labels)
            print(f"Data for {action} included.")
        else:
            print(f"Data for {action} discarded.")
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
    print("Data collection appended and saved for all actions.")

# ----------------------------
# Graphical User Interface (GUI)
# ----------------------------
def start_gui():
    """
    Builds and runs the main GUI.
    Displays real-time status for EEG mode, classifier type, and model statistics.
    Provides buttons for data collection, model training, starting/stopping EEG processing,
    toggling modes, refreshing stats, and clean exit.
    """
    global running, SIMULATED, USE_DEEP_LEARNING, model_stats_var, simulated_mode_var, classifier_mode_var

    root = tk.Tk()
    root.title("EEG Keyboard Control System")

    # StringVars for dynamic status display
    simulated_mode_var = tk.StringVar(root)
    simulated_mode_var.set("Simulated EEG Mode" if SIMULATED else "Real EEG Mode")
    classifier_mode_var = tk.StringVar(root)
    classifier_mode_var.set("Deep Learning" if USE_DEEP_LEARNING else "SVM")
    model_stats_var = tk.StringVar(root)
    model_stats_var.set("No model stats available.")

    # Status labels
    sim_label = tk.Label(root, textvariable=simulated_mode_var, font=("Helvetica", 10))
    sim_label.pack(pady=2)
    class_label = tk.Label(root, textvariable=classifier_mode_var, font=("Helvetica", 10))
    class_label.pack(pady=2)
    stats_label = tk.Label(root, textvariable=model_stats_var, font=("Helvetica", 10), justify=tk.LEFT)
    stats_label.pack(pady=2)

    # Button callback functions
    def on_start_collection():
        threading.Thread(target=start_collection).start()

    def on_train_model():
        threading.Thread(target=train_model).start()
        show_model_stats()

    def on_start_eeg():
        global running
        running = True
        threading.Thread(target=process_eeg_commands).start()
        threading.Thread(target=visualize_eeg).start()

    def on_stop_eeg():
        global running
        running = False

    def on_toggle_simulated():
        global SIMULATED
        SIMULATED = not SIMULATED
        status = "Simulated EEG Mode" if SIMULATED else "Real EEG Mode"
        simulated_mode_var.set(status)
        print(status)

    def on_toggle_classifier():
        global USE_DEEP_LEARNING
        USE_DEEP_LEARNING = not USE_DEEP_LEARNING
        mode = "Deep Learning" if USE_DEEP_LEARNING else "SVM"
        classifier_mode_var.set(mode)
        print(f"Classifier set to {mode}")

    # Create GUI buttons
    collection_button = tk.Button(root, text="Start Collection", command=on_start_collection)
    collection_button.pack(pady=5)
    train_button = tk.Button(root, text="Train Model", command=on_train_model)
    train_button.pack(pady=5)
    start_button = tk.Button(root, text="Start EEG Processing", command=on_start_eeg)
    start_button.pack(pady=5)
    stop_button = tk.Button(root, text="Stop EEG Processing", command=on_stop_eeg)
    stop_button.pack(pady=5)
    stats_button = tk.Button(root, text="Refresh Model Stats", command=show_model_stats)
    stats_button.pack(pady=5)
    toggle_simulated_button = tk.Button(root, text="Toggle Simulated/Real EEG", command=on_toggle_simulated)
    toggle_simulated_button.pack(pady=5)
    toggle_classifier_button = tk.Button(root, text="Toggle Classifier (SVM/DL)", command=on_toggle_classifier)
    toggle_classifier_button.pack(pady=5)

    def on_exit():
        global running
        running = False
        close_board()
        root.destroy()

    exit_button = tk.Button(root, text="Exit", command=on_exit)
    exit_button.pack(pady=5)
    root.protocol("WM_DELETE_WINDOW", on_exit)
    root.mainloop()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Ensure necessary directories exist
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("data"):
        os.makedirs("data")
    # Initialize the BrainFlow board if not using simulated data
    if not SIMULATED:
        initialize_board()
    # Attempt to load an existing model (if available)
    load_model()
    # Start the GUI
    start_gui()
