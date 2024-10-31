import numpy as np
import pandas as pd
import tensorflow as tf
import keras 
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector, Conv1D, BatchNormalization
from tensorflow.keras.losses import MeanAbsoluteError
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# Algorithm for calculating Root Mean Square Error for a single feature
@keras.saving.register_keras_serializable()
def metric_rmse_for_feature(y_true, y_pred):
    return tf.sqrt(metric_mse_for_feature(y_true, y_pred))

# Algorithm for calculating Mean Square Error for a single feature
@keras.saving.register_keras_serializable()
def metric_mse_for_feature(y_true, y_pred):
    y_true_feature = y_true[:, :, 0]
    y_pred_feature = y_pred[:, :, 0]
    
    mse = tf.reduce_mean(tf.square(y_true_feature - y_pred_feature))
    return mse

# Algorithm for calculating Symmetric Mean Absolute Percentage Error
@keras.saving.register_keras_serializable()
def metric_smape(y_true, y_pred):
    epsilon = 1e-10 
    diff = tf.abs(y_true - y_pred)
    scale = tf.abs(y_true) + tf.abs(y_pred) + epsilon
    return 200 * tf.reduce_mean(diff / scale)

# Algorithm for calculating Coefficient of Determination
@keras.saving.register_keras_serializable()
def metric_r2score(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

class StockModel:

    def __init__(self, frame: pd.DataFrame, steps: int = 30, horizon: int = 10):
        # ====================================================
        # Configuration
        # ====================================================
        self.steps = steps      # The historically observed datapoints (days)
        self.horizon = horizon  # The number of future datapoints (days) to predict
        self.learning_rate = 0.001 # The learning rate of the model
        self.epocs = 100
        self.batchsize = 64
        self.units = 50
        self.name = frame.values[0][0] # The unique name of the symbol

        # ====================================================
        # The trained model
        # ====================================================
        # Either trained on the fly or loaded from disk
        self.model = None

        # ====================================================
        # The scaler to be used
        # ====================================================
        # It is important that the scalers configuration is serialized
        # together with the model as it needs to be configured
        # with the training data. If the model is loaded from
        # file. The scaling needs to be restored separately.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # ====================================================
        # Extract fetures to train the model with
        # ====================================================
        # The first column which represents the Closing price
        # is to be considered the conclusive output for the
        # prediction (The "label"). All 5 features will be 
        # used in the trining.
        self.dataset = frame[[
            'Close',
            'Adj Close', 
            'Open', 
            'High', 
            'Low', 
            'Volume', 
            'SMA_high', 'SMA_low',
            'EMA_high', 'EMA_low'
            ]]
        
        self.features = len(self.dataset.columns)
        
    def train_model(self) -> Sequential:
        # ====================================================
        # Fit the transform to training data 
        # ====================================================
        # The frame will be flattened but the shape will still 
        # be (samples, features)
        scaled_dataset = self.scaler.fit_transform(
            self.dataset)
        
        # ====================================================
        # Create timeseries for training 
        # ====================================================
        # The shape of the return values constitutes:
        # x: (samples, timesteps, feature_count)
        # y: (samples, horizon, feature_count)       
        x, y = self.create_sequences(
            scaled_dataset)

        # ====================================================
        # Split data between training and validation data
        # ====================================================
        # A common split is 80/20 which is also used here
        size = int(len(x) * 0.8)

        x_train, x_val = x[:size], x[size:]
        y_train, y_val = y[:size], y[size:]

        # ====================================================
        # Create pipeline
        # ====================================================
        self.model = Sequential([
            Input(shape=(self.steps, self.features)),
            LSTM(self.units, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(self.units, activation='relu', return_sequences=False),
            Dropout(0.2),
            RepeatVector(self.horizon),
            LSTM(self.units, activation='relu', return_sequences=True),
            Dropout(0.2),
            TimeDistributed(Dense(self.features))
        ])
        # ====================================================
        # Compile model
        # ====================================================
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
            loss=MeanAbsoluteError(), 
            metrics=[
                'mse',
                metric_rmse_for_feature,
                metric_mse_for_feature,
                metric_smape,
                metric_r2score])
    
        # ====================================================
        # Create stopping function to avoid overfitting
        # ====================================================
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True)

        # ====================================================
        # Train the model
        # ====================================================
        result = self.model.fit(
            x_train, 
            y_train, 
            epochs=self.epocs, 
            batch_size=self.batchsize, 
            validation_data=(x_val, y_val),
            callbacks=[early_stopping],
            verbose=2)
        
        # ====================================================
        # Make predictions to compare with the validation data
        # ====================================================
        predictions = self.model.predict(x_val)

        # ====================================================
        # Reshape data
        # ====================================================
        rescaled_predictions = self.scaler.inverse_transform(predictions
            .reshape(-1, predictions.shape[2])).reshape(predictions.shape)
        rescaled_validation = self.scaler.inverse_transform(y_val
            .reshape(-1, y_val.shape[2])).reshape(y_val.shape)
        
        self.plot_metrics(
            result, 
            rescaled_predictions, 
            rescaled_validation)
        return self.model

    def create_sequences(self, data: np.array) -> tuple[np.array, np.array]:
        # ====================================================
        # Timeseries fit for LSTM
        # ====================================================
        x, y = [], []

        for i in range(len(data) - self.steps - self.horizon +1):
            x.append(data[i:i + self.steps])
            y.append(data[i + self.steps: i + self.steps + self.horizon])

        return np.array(x), np.array(y)

    def plot_metrics(self, results: any, predictions: pd.array, validation: pd.array):
        predictions_close = predictions[:, -1, 0]
        validation_close = validation[:, -1, 0]

        fig = plt.figure(figsize=(20, 10), layout="constrained")
        spec = fig.add_gridspec(3, 3)

        ax = fig.add_subplot(spec[0, :])
        ax.plot(validation_close, label="True Values")
        ax.plot(predictions_close, label="Predictions")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(True)
        ax.legend()

        ax = fig.add_subplot(spec[1, 0])
        ax.set_title("Loss")
        ax.plot(results.epoch, results.history['loss'], label=f'Train: {np.mean(results.history['loss'])}')
        ax.plot(results.epoch, results.history['val_loss'], label=f'Validation: {np.mean(results.history['val_loss'])}')
        ax.legend()

        ax = fig.add_subplot(spec[1, 1])
        ax.set_title("Root Mean Square Error (Close)")
        ax.plot(results.epoch, results.history['metric_rmse_for_feature'], label=f'RMSE: {np.mean(results.history['metric_rmse_for_feature'])}')
        ax.legend()

        ax = fig.add_subplot(spec[1, 2])
        ax.set_title("Mean Square Error")
        ax.plot(results.epoch, results.history['metric_mse_for_feature'], label=f'Close: {np.mean(results.history['metric_mse_for_feature'])}')
        ax.plot(results.epoch, results.history['mse'], label=f'Train: {np.mean(results.history['mse'])}')
        ax.plot(results.epoch, results.history['val_mse'], label=f'Validation: {np.mean(results.history['val_mse'])}')
        ax.legend()

        ax = fig.add_subplot(spec[2, 0])
        ax.set_title("Symmetric Mean Absolute Percentage Error")
        ax.plot(results.epoch, results.history['metric_smape'], label=f'sMAPE Train: {np.mean(results.history['metric_smape'])}')
        ax.plot(results.epoch, results.history['val_metric_smape'], label=f'sMAPE Validation: {np.mean(results.history['val_metric_smape'])}')
        ax.legend()

        ax = fig.add_subplot(spec[2, 1])
        ax.set_title("Coefficient of Determination")
        ax.plot(results.epoch, results.history['metric_r2score'], label=f'R2 Train: {np.mean(results.history['metric_r2score'])}')
        ax.plot(results.epoch, results.history['val_metric_r2score'], label=f'R2 Validation: {np.mean(results.history['val_metric_r2score'])}')
        ax.legend()

        plt.show()

    def predict(self) -> pd.DataFrame:
        # ===========================================
        # Scale to fit 0 - 1
        # ===========================================
        # Extract the last N days from the most recent dataset
        scaled_dataset = self.scaler.transform(self.dataset[-self.steps:])

        # ===========================================
        # Reshape scaled data
        # ===========================================
        # Shape needed is: (Samples, steps, features)
        reshaped_dataset = scaled_dataset.reshape(
            -1, 
            self.steps ,
            self.features)
        
        # ===========================================
        # Perform prediction
        # ===========================================
        predictions = self.model.predict(reshaped_dataset)

        # ===========================================
        # Reshape predictions 
        # ===========================================
        # Shape needed is: (days, features)
        reshape = predictions.reshape(-1, len(self.dataset.columns))
        
        # ===========================================
        # Inverse transform
        # ===========================================        
        inverse = self.scaler.inverse_transform(reshape)

        # ===========================================
        # Create a date range in the future
        # =========================================== 
        # This way a dataframe is built matching
        # the original dataset closely.
        # Using the last day in the dataset +1 to be
        # the first day of prediction
        index = pd.date_range(self.dataset.index[-1], periods=len(inverse)+1)[1:]
        # Populate Dataframe
        frame = pd.DataFrame({
            'Date': index,
            'Close': inverse[:,0],
            'Open': inverse[:,1],
            'High': inverse[:,2],
            'Low': inverse[:,3],
            'Volume':inverse[:,4],
        })
        # Set index 
        return frame.set_index("Date")

    def fromFile(self):
        self.model = keras.saving.load_model(f'../models/{self.name}.keras')
        self.scaler = joblib.load(f'../models/{self.name}.save')
        
    def toFile(self):
        if self.model != None:
            self.model.save(f'../models/{self.name}.keras')
            joblib.dump(self.scaler, f'../models/{self.name}.save')

