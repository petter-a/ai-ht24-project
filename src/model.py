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
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector
from tensorflow.keras.losses import MeanAbsoluteError
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

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

    def __init__(self, frame: pd.DataFrame):
        # ====================================================
        # Configuration
        # ====================================================
        self.steps = 30         # The historically observed datapoints (days)
        self.horizon = 10       # The number of future datapoints (days) to predict
        self.learning_rate = 0.001 # The learning rate of the model
        self.epocs = 100
        self.batchsize = 32
        self.units = 50
        self.patience = 8
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
        self.scaler = MinMaxScaler(
            feature_range=(0, 1))
        
        # ====================================================
        # Extract fetures to train the model with
        # ====================================================
        # The first column which represents the Closing price
        # is to be considered the conclusive output for the
        # prediction (The "label"). All 5 features will be 
        # used in the trining.
        self.dataset = frame[[
            'Adj Close', 
            'Volume',
            'SMA_high', 'SMA_low',
            'EMA_high', 'EMA_low'
            ]]
        
        self.features = len(self.dataset.columns)
        
    def train_model(self, interactive: bool = True) -> Sequential:
        # ====================================================
        # Fit the transform to training data 
        # ====================================================
        # The frame will be flattened but the shape will still 
        # be (samples, features)
        scaled_dataset = self.scaler.fit_transform(
            self.dataset)
        
        # ====================================================
        # Split data between training and validation data
        # ====================================================
        # A common split is 80/20 which is also used here
        size = int(len(scaled_dataset) * 0.8)
        train_set, valid_set = scaled_dataset[:size], scaled_dataset[size:]

        # ====================================================
        # Create timeseries for training 
        # ====================================================
        # The shape of the return values constitutes:
        # x: (samples, timesteps, feature_count)
        # y: (samples, horizon, feature_count) 
        x_train, y_train = self.create_sequences(train_set)
        x_val, y_val = self.create_sequences(valid_set)
        
        # ====================================================
        # Create pipeline
        # ====================================================
        self.model = Sequential([
            Input(shape=(self.steps, self.features)),
            LSTM(self.units, activation='relu'),
            Dropout(0.2),
            RepeatVector(self.horizon),
            LSTM(self.units, activation='relu', return_sequences=True),
            TimeDistributed(Dense(self.features))
        ])
        # ====================================================
        # Compile model
        # ====================================================
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
            loss=MeanAbsoluteError(), 
            metrics=[
                'mse',
                metric_smape,
                metric_r2score])
    
        # ====================================================
        # Create stopping function to avoid overfitting
        # ====================================================
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=self.patience, 
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

        if interactive:
            # ====================================================
            # Reshape data
            # ====================================================
            rescaled_predictions = self.scaler.inverse_transform(predictions
                .reshape(-1, predictions.shape[2])).reshape(predictions.shape)
            rescaled_validations = self.scaler.inverse_transform(valid_set)
            
            self.plot_metrics(
                result, 
                rescaled_predictions, 
                rescaled_validations)

        return self.model

    def create_sequences(self, data: np.array) -> tuple[np.array, np.array]:
        # ====================================================
        # Timeseries fit for LSTM
        # ====================================================
        x, y = [], []

        for i in range(len(data) - self.steps - self.horizon +1):
            x.append(data[i:i + self.steps])
            y.append(data[i + self.steps:i + self.steps + self.horizon])

        return np.array(x), np.array(y)

    def plot_metrics(self, results: any, predictions: pd.array, validation: pd.array):
        predictions_close = predictions[:, :, 0].mean(axis=1)
        validation_close = validation[:,0]

        fig = plt.figure(figsize=(20, 10), layout="constrained")
        spec = fig.add_gridspec(3, 3)

        ax = fig.add_subplot(spec[0, :])
        ax.set_title(self.name)
        ax.plot(validation_close, label="True Values")
        ax.plot(predictions_close, label="Predictions")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(True)
        ax.legend()

        ax = fig.add_subplot(spec[1, 0])
        ax.set_title("Loss (MAE)")
        ax.plot(results.epoch, results.history['loss'], label=f'Train: {np.mean(results.history['loss']):.4f}')
        ax.plot(results.epoch, results.history['val_loss'], label=f'Validation: {np.mean(results.history['val_loss']):.4f}')
        ax.legend()

        ax = fig.add_subplot(spec[1, 1])
        ax.set_title("Root Mean Square Error")
        ax.plot(results.epoch, tf.sqrt(results.history['mse']), label=f'Train: {np.mean(tf.sqrt(results.history['mse'])):.4f}')
        ax.plot(results.epoch, tf.sqrt(results.history['val_mse']), label=f'Validation: {np.mean(tf.sqrt(results.history['val_mse'])):.4f}')
        ax.legend()

        ax = fig.add_subplot(spec[1, 2])
        ax.set_title("Mean Square Error")
        ax.plot(results.epoch, results.history['mse'], label=f'Train: {np.mean(results.history['mse']):.4f}')
        ax.plot(results.epoch, results.history['val_mse'], label=f'Validation: {np.mean(results.history['val_mse']):.4f}')
        ax.legend()

        ax = fig.add_subplot(spec[2, 0])
        ax.set_title("Symmetric Mean Absolute Percentage Error")
        ax.plot(results.epoch, results.history['metric_smape'], label=f'Train: {np.mean(results.history['metric_smape']):.4f}')
        ax.plot(results.epoch, results.history['val_metric_smape'], label=f'Validation: {np.mean(results.history['val_metric_smape']):.4f}')
        ax.legend()

        ax = fig.add_subplot(spec[2, 1])
        ax.set_title("Coefficient of Determination")
        ax.plot(results.epoch, results.history['metric_r2score'], label=f'Train: {np.mean(results.history['metric_r2score']):.4f}')
        ax.plot(results.epoch, results.history['val_metric_r2score'], label=f'Validation: {np.mean(results.history['val_metric_r2score']):.4f}')
        ax.legend()

        plt.show()

    def predict(self) -> pd.DataFrame:
        # ===========================================
        # Scale to fit 0 - 1
        # ===========================================
        # Extract the last N days from the most recent dataset
        scaled_dataset = self.scaler.transform(
            self.dataset[-(self.steps*2):])

        # ===========================================
        # Reshape scaled data
        # ===========================================
        # Shape needed is: (Samples, steps, features)
        x_predict, _ = self.create_sequences(scaled_dataset)
        
        # ===========================================
        # Perform prediction
        # ===========================================
        predictions = self.model.predict(x_predict)

        # ===========================================
        # Reshape predictions 
        # ===========================================
        # Shape needed is: (days, features)
        rescaled_predictions = self.scaler.inverse_transform(predictions
            .reshape(-1, predictions.shape[2])).reshape(predictions.shape)

        # ===========================================
        # Create a date range in the future
        # =========================================== 
        # This way a dataframe is built matching
        # the original dataset closely.
        # Using the last day in the dataset +1 to be
        # the first day of prediction
        closing_price = rescaled_predictions[:, :, 0]
       
        index = pd.date_range(self.dataset.index[-1], periods=len(closing_price)+1)[1:]
        # Populate Dataframe
        frame = pd.DataFrame(rescaled_predictions[:, :, 0].mean(axis=1), columns=['Adj Close'])
        # Set index 
        return frame.set_index(index)

    def fromFile(self):
        self.model = keras.saving.load_model(f'../models/{self.name}.keras')
        self.scaler = joblib.load(f'../models/{self.name}.save')
        
    def toFile(self):
        if self.model != None:
            self.model.save(f'../models/{self.name}.keras')
            joblib.dump(self.scaler, f'../models/{self.name}.save')

