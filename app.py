import os
from pathlib import Path
import numpy as np
import librosa
import pywt # For Wavelet operations
from scipy.stats import entropy
from tensorflow.keras.models import load_model # Ensure this matches your TF version
import joblib # For loading the scaler
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import uuid # For generating unique filenames
import logging

# --- Application Setup ---
app = Flask(__name__)
# Secret key is needed for flashing messages to the user
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads" # Temporary storage for uploaded files
MODEL_DIR = BASE_DIR / "Model"       # Directory where model and scaler are stored

# --- IMPORTANT: UPDATE THESE PATHS IF YOUR FILENAMES ARE DIFFERENT ---
MODEL_FILENAME = "audio_model.h5" 
SCALER_FILENAME = "scaler.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
SCALER_PATH = MODEL_DIR / SCALER_FILENAME
# --- END OF PATHS TO UPDATE ---

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15MB limit for audio files

# --- Logging Setup ---
# Configure logging to provide insights into the application's behavior
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Load Model and Scaler ---
# This section attempts to load the pre-trained model and scaler.
# Errors here will prevent the core functionality of the app.
model = None
scaler = None
try:
    if not MODEL_PATH.exists():
        logging.error(f"CRITICAL: Model file not found at {MODEL_PATH}. Please ensure it's in the Model/ directory.")
    else:
        model = load_model(MODEL_PATH) # Keras model loading
        logging.info(f"Successfully loaded model from {MODEL_PATH}")
    
    if not SCALER_PATH.exists():
        logging.error(f"CRITICAL: Scaler file not found at {SCALER_PATH}. Please ensure it's in the Model/ directory.")
    else:
        scaler = joblib.load(SCALER_PATH) # Scikit-learn scaler loading
        logging.info(f"Successfully loaded scaler from {SCALER_PATH}")
    
    if not model or not scaler:
        logging.critical("Model or scaler failed to load. The application will not function correctly.")
        # In a real production app, you might have a fallback or maintenance mode.
except Exception as e:
    logging.critical(f"Fatal error during model or scaler loading: {e}", exc_info=True)
    # Ensure model and scaler are None if loading fails
    model = None
    scaler = None


# --- Feature Extraction Parameters ---
# These parameters must match those used during model training.
N_FEATURES = 41       # Expected total number of features
TARGET_SR = 16000     # Target sample rate for audio processing
EWT_WAVELET = 'db4'   # Wavelet type for EWT (DWT approximation)
EWT_LEVEL = 4         # Decomposition level for EWT
WPT_WAVELET = 'db4'   # Wavelet type for WPT
WPT_LEVEL = 3         # Decomposition level for WPT


# --- Feature Extraction Functions ---

def load_and_preprocess_audio(file_path, target_sr=TARGET_SR):
    """Loads an audio file, resamples to target_sr, converts to mono, and normalizes."""
    try:
        # Load audio file, resample to target_sr, and convert to mono
        y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        
        # Normalize audio signal to range [-1, 1]
        max_abs_val = np.max(np.abs(y))
        if max_abs_val > 1e-6: # Avoid division by zero for silent or near-silent audio
            y_normalized = y / max_abs_val
        else:
            y_normalized = y # Keep as is if silent
            logging.warning(f"Audio file {Path(file_path).name} appears to be silent or near-silent.")
        
        # Ensure audio is not excessively short (e.g., less than 100ms)
        min_duration_samples = int(target_sr * 0.1) # 100ms
        if len(y_normalized) < min_duration_samples:
            logging.warning(f"Audio file {Path(file_path).name} is very short ({len(y_normalized)/target_sr:.2f}s). Padding with zeros.")
            y_normalized = np.pad(y_normalized, (0, min_duration_samples - len(y_normalized)), 'constant')
            
        return y_normalized, target_sr
    except Exception as e:
        logging.error(f"Error loading or preprocessing audio file {Path(file_path).name}: {e}", exc_info=True)
        return None, None

def ewt_decompose(signal, wavelet=EWT_WAVELET, level=EWT_LEVEL):
    """Approximates Empirical Wavelet Transform using Discrete Wavelet Transform (DWT)."""
    # Returns a list of coefficient arrays (approximation at level N, then details from N to 1)
    return pywt.wavedec(signal, wavelet, level=level, mode='symmetric')

def wpt_decompose(signal, wavelet=WPT_WAVELET, level=WPT_LEVEL):
    """Performs Wavelet Packet Transform."""
    # Creates a WaveletPacket object
    wp_obj = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level, mode='symmetric')
    # Gets coefficient arrays for all nodes at the specified level in natural order
    nodes_at_level = [node.path for node in wp_obj.get_level(level, 'natural')]
    return [wp_obj[node].data for node in nodes_at_level]

def calculate_subband_features(coeffs_list):
    """Calculates mean, variance, and entropy for each list of subband coefficients."""
    features = []
    for coeffs in coeffs_list:
        if coeffs is not None and len(coeffs) > 0:
            features.extend([
                np.mean(coeffs),
                np.var(coeffs),
                entropy(np.abs(coeffs) + 1e-10) # Epsilon for numerical stability with entropy
            ])
        else: # Handle cases of empty or None coefficient arrays
            features.extend([0.0, 0.0, 0.0]) # Append placeholder zero-features
    return features

def extract_spectral_feature(y, sr):
    """Extracts a single spectral feature (e.g., mean spectral flatness)."""
    if y is None or len(y) == 0: return 0.0
    # Spectral flatness measures the "evenness" of the power spectrum
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    return np.mean(spectral_flatness) if spectral_flatness is not None else 0.0

def extract_pitch_related_feature(y, sr):
    """Extracts a single pitch-related feature (e.g., standard deviation of fundamental frequency)."""
    if y is None or len(y) == 0: return 0.0
    # YIN algorithm for fundamental frequency (F0) estimation
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_valid = f0[~np.isnan(f0)] # Filter out NaN values from F0 estimation
    if len(f0_valid) > 1: # Need at least two valid F0 values to calculate std dev
        return np.std(f0_valid)
    return 0.0 # Return 0 if not enough valid F0 values

def extract_all_features(file_path):
    """Main function to extract all 41 features from an audio file."""
    y, sr = load_and_preprocess_audio(file_path)
    if y is None:
        logging.error(f"Feature extraction failed: Could not load/preprocess audio from {Path(file_path).name}")
        return None

    try:
        # EWT features: (level + 1) sets of coeffs * 3 features/set
        # For EWT_LEVEL=4, this is 5 sets * 3 = 15 features
        ewt_coeffs_list = ewt_decompose(y)
        ewt_features = calculate_subband_features(ewt_coeffs_list)

        # WPT features: 2^level sets of coeffs * 3 features/set
        # For WPT_LEVEL=3, this is 2^3 = 8 sets * 3 = 24 features
        wpt_coeffs_list = wpt_decompose(y)
        wpt_features = calculate_subband_features(wpt_coeffs_list)
        
        # Single spectral feature
        spectral_feat = extract_spectral_feature(y, sr)
        
        # Single pitch-related feature
        pitch_feat = extract_pitch_related_feature(y, sr)

        # Combine all features in the order expected by the model
        # Order: [spectral_feat (1), pitch_feat (1), ewt_features (15), wpt_features (24)]
        combined_features_list = [spectral_feat, pitch_feat] + ewt_features + wpt_features
        
        # Convert to NumPy array and handle potential NaNs/Infs robustly
        feature_vector_np = np.array(combined_features_list, dtype=np.float32)
        feature_vector_np = np.nan_to_num(feature_vector_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        final_feature_vector = feature_vector_np.reshape(1, -1) # Reshape for model input

        # Critical check for feature count
        if final_feature_vector.shape[1] != N_FEATURES:
            logging.error(f"CRITICAL Feature mismatch for {Path(file_path).name}: Extracted {final_feature_vector.shape[1]} features, but model expects {N_FEATURES}.")
            return None 

        logging.info(f"Successfully extracted {final_feature_vector.shape[1]} features for {Path(file_path).name}.")
        return final_feature_vector

    except Exception as e:
        logging.error(f"Error during feature extraction pipeline for {Path(file_path).name}: {e}", exc_info=True)
        return None

# --- Flask Routes ---

@app.route("/", methods=["GET"])
def index():
    """Renders the main page."""
    return render_template("index.html", prediction_label=None, confidence_score=None)

@app.route("/predict", methods=["POST"])
def predict():
    """Handles audio file upload and prediction."""
    # Check if model and scaler are loaded; if not, app cannot function.
    if not model or not scaler:
        flash("Application Error: Model or scaler is not loaded. Please contact support or check server logs.", "danger")
        logging.critical("Attempted prediction while model/scaler not loaded.")
        return redirect(url_for('index'))

    # Validate file presence in request
    if 'audio_file' not in request.files:
        flash("No audio file part found in the request. Please select a file.", "warning")
        return redirect(url_for('index'))

    file = request.files['audio_file']
    if file.filename == '':
        flash("No audio file selected for upload. Please choose a file.", "warning")
        return redirect(url_for('index'))

    # Process the file if it exists
    if file:
        # Secure the filename and generate a unique name for temporary storage
        original_filename = secure_filename(file.filename)
        if not original_filename: # Handle cases where secure_filename might return an empty string
            flash("Invalid filename. Please use standard characters in the filename.", "danger")
            return redirect(url_for('index'))
            
        unique_id = uuid.uuid4().hex
        # Use a combination of UUID and original extension for the saved file
        _, ext = os.path.splitext(original_filename)
        saved_filename = f"{unique_id}{ext}" if ext else f"{unique_id}.unknown" # Handle missing extension
        
        filepath = UPLOAD_FOLDER / saved_filename
        
        try:
            file.save(filepath) # Save uploaded file temporarily
            logging.info(f"File '{original_filename}' saved temporarily as '{saved_filename}' for processing.")

            # Extract features from the saved audio file
            features = extract_all_features(filepath)

            if features is None:
                flash("Could not extract features from the audio file. It might be corrupted, silent, too short, or in an unsupported format. Check logs for details.", "danger")
                return redirect(url_for('index'))

            # Scale features using the loaded scaler
            scaled_features = scaler.transform(features)
            
            # Make prediction using the loaded model
            # Assuming model.predict() returns a probability for the positive class (e.g., "Real Audio")
            prediction_prob = model.predict(scaled_features)[0][0] 

            # Determine label and confidence based on a threshold
            threshold = 0.5 
            if prediction_prob >= threshold:
                label = "Real Audio"
                confidence = prediction_prob * 100
            else:
                label = "Fake Audio"
                confidence = (1 - prediction_prob) * 100 # Confidence for the "Fake" class
            
            logging.info(f"Prediction for '{original_filename}': {label} with {confidence:.2f}% confidence (Raw prob: {prediction_prob:.4f}).")
            
            # Render the page again with prediction results
            return render_template("index.html", prediction_label=label, confidence_score=f"{confidence:.2f}")

        except Exception as e:
            logging.error(f"An unexpected error occurred during prediction for '{original_filename}': {e}", exc_info=True)
            flash(f"An unexpected error occurred while processing the file. Please try again or check server logs.", "danger")
            return redirect(url_for('index'))
        finally:
            # Clean up: remove the temporarily saved audio file
            if filepath.exists():
                try:
                    os.remove(filepath)
                    logging.info(f"Successfully removed temporary file: {filepath.name}")
                except Exception as e_remove:
                    logging.error(f"Error removing temporary file {filepath.name}: {e_remove}", exc_info=True)
    
    # Fallback redirect if 'file' object is not valid for some reason
    return redirect(url_for('index'))

# --- Main Execution ---
if __name__ == "__main__":
    # This block runs when the script is executed directly (e.g., `python app.py`)
    # For production, use a WSGI server like Gunicorn (specified in Procfile).
    # The host '0.0.0.0' makes the app accessible from other devices on the network.
    # Debug mode should be False in production.
    is_debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=is_debug_mode)
