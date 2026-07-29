import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

class BiosignalPredictor:
    def __init__(self, model_path="model.tflite"):
        """Initializes the ML Predictor using a compiled TensorFlow Lite model."""
        self.interpreter = None
        self.N_MELS = 64
        self.MAX_TIME_STEPS = 157  
        
        if TFLITE_AVAILABLE and LIBROSA_AVAILABLE:
            try:
                import os
                if not os.path.exists(model_path):
                    # We expect it to be missing on a brand new setup, avoid crash
                    raise FileNotFoundError("Model file does not exist locally.")
                    
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                # Do nothing, this is expected before the user runs `train_ml_model.py`
                pass
        else:
            print("[ML Notice] TensorFlow or Librosa not found. ML functionality disabled.")

    def extract_features(self, audio_window, fs):
        """Converts the real-time audio window into the same Spectrogram format used during training."""
        # Convert window to float array
        y = np.array(audio_window, dtype=np.float32)
        
        # 1. Generate Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=self.N_MELS, fmax=fs/2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 2. Match Time Steps length
        if S_dB.shape[1] < self.MAX_TIME_STEPS:
            pad_width = self.MAX_TIME_STEPS - S_dB.shape[1]
            S_dB = np.pad(S_dB, pad_width=((0,0), (0, pad_width)), mode='constant')
        else:
            S_dB = S_dB[:, :self.MAX_TIME_STEPS]
            
        # 3. Add Channel dimension: [N_MELS, TIME, 1]
        features = np.expand_dims(S_dB, axis=-1)
        
        # 4. Add Batch dimension required by TFLite: [1, N_MELS, TIME, 1]
        features = np.expand_dims(features, axis=0) 
        
        return features.astype(np.float32)

    def predict(self, audio_data, fs):
        """Run ML model to predict Heart Rate and Respiratory Rate."""
        if not self.interpreter:
            return None, None
            
        try:
            # 1. Feature Extraction
            features = self.extract_features(audio_data, fs)
            
            # 2. Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            self.interpreter.invoke()
            
            # 3. Results Decode 
            # output shape was designed as a dense layer with 2 elements: [HR, RR]
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            hr_pred = output_data[0][0]
            rr_pred = output_data[0][1]
            
            return float(hr_pred), float(rr_pred)
            
        except Exception as e:
            # Silently fail if something goes catastrophically wrong with tensor sizes
            return None, None
