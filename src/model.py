import numpy as np
import pandas as pd
import tensorflow as tf
import keras 
import joblib
import config
from typing import Self
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
        self.train_size = 0.7
        self.valid_size = 0.15
        self.steps = 20         # The historically observed datapoints (days)
        self.horizon = 1        # The number of future datapoints to predict
        self.learning_rate = 0.001 # The learning rate of the model
        self.epocs = 100
        self.batchsize = self.steps
        self.units = 50
        self.patience = 4
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
            'EMA_high', 'EMA_low',
            'RSI_val'
            ]]
        
        self.features = len(self.dataset.columns)
        
    def train_model(self, interactive: bool = True) -> Self:
        # ====================================================
        # Fit the transform to training data 
        # ====================================================

        scaled_dataset = self.scaler.fit_transform(
            self.dataset)
        
        '''
        Example input dataset. Shape: (Samples, Features):
        
        print(self.dataset.tail(3))

                     Adj Close     Volume    SMA_high     SMA_low    EMA_high     EMA_low
        Date                                                                             
        2024-10-30  127.559998  2737200.0  104.462501  132.911551  131.504459  130.393407
        2024-10-31  128.470001  3408800.0  104.673950  132.910209  131.340434  130.097498
        2024-11-01  127.220001  3068700.0  104.877687  132.896400  131.117708  129.654806

        Example scaled data:

        print(self.scaler.fit_transform(self.dataset.tail(3)))

        [[0.27199707 0.         0.         1.         1.         1.        ]
        [ 1.         1.         0.50928784 0.91145309 0.57589041 0.59936585]
        [ 0.         0.49359738 1.         0.         0.         0.        ]]
                
        '''
        
        # ====================================================
        # Split data between training, validation and test
        # ====================================================
        # 70% used for training, 
        # 15% for validation
        # Remaining data for test 

        train_size = int(len(scaled_dataset) * self.train_size)
        valid_size = int(len(scaled_dataset) * self.valid_size + train_size)

        train_set, valid_set, tests_set = (
            scaled_dataset[:train_size],
            scaled_dataset[train_size: valid_size],
            scaled_dataset[valid_size:]
        )

        '''
        print(len(train_set))
        2613

        print(len(valid_set))
        560

        print(len(tests_set))
        561
        '''
        # ====================================================
        # Create timeseries for training 
        # ====================================================
        # The shape of the return values constitutes:
        #
        # x: (samples, timesteps, features)
        # y: (samples, horizon, features) 
        # 
        x_train, y_train = self.create_sequences(train_set)
        x_valid, y_valid = self.create_sequences(valid_set)
        x_tests, y_tests = self.create_sequences(tests_set)
        
        # ====================================================
        # Create LSTM pipeline
        # ====================================================
        # RepeatVector is used for MultiStep prediction
        self.model = Sequential([
            Input(shape=(self.steps, self.features)),
            LSTM(self.units, activation='relu'),
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
            validation_data=(x_valid, y_valid),
            callbacks=[early_stopping],
            verbose=2)
        
        # ====================================================
        # Evaluate model
        # ====================================================
        eval_metrics = self.model.evaluate(x_tests, y_tests)
        
        for i, metric in enumerate(eval_metrics):
            print(f'Metric-{i}: {metric}')
                
        if interactive:
            # ====================================================
            # Make predictions on validation data
            # ====================================================
            predictions = self.model.predict(x_valid)

            # ====================================================
            # Reshape data
            # ====================================================
            
            # Restore predictions
            rescaled_predictions = self.scaler.inverse_transform(predictions
                .reshape(-1, predictions.shape[2])).reshape(predictions.shape)
            
            # Restore validation set
            rescaled_validations = self.scaler.inverse_transform(y_valid
                .reshape(-1, y_valid.shape[2])).reshape(y_valid.shape)
            
            self.plot_metrics(
                result,
                [len(train_set), len(valid_set), len(tests_set)],
                eval_metrics,
                rescaled_predictions, 
                rescaled_validations)

        return self

    def create_sequences(self, data: np.array) -> tuple[np.array, np.array]:
        # ====================================================
        # Timeseries fit for LSTM
        # ====================================================
        x, y = [], []

        for i in range(len(data) - self.steps - self.horizon +1):
            x.append(data[i:i + self.steps])
            y.append(data[i + self.steps:i + self.steps + self.horizon])

        return np.array(x), np.array(y)

    def predict(self) -> pd.DataFrame:
        # ===========================================
        # Scale to fit 0 - 1
        # ===========================================
        # Extract the last N days from the most recent dataset
        scaled_dataset = self.scaler.transform(
            self.dataset[-(self.steps*3):])

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
        prices = rescaled_predictions[:, :, 0]
        volume = rescaled_predictions[:, :, 1]

        index = pd.date_range(self.dataset.index[-1], periods=rescaled_predictions.shape[0]+1)[1:]
        # Populate Dataframe
        frame = pd.DataFrame({
            'Adj Close': prices[:,0],
            'Volume': volume[:,0]
            })
        # Set index 
        return frame.set_index(index)
    
    def plot_metrics(self, results: any, sizes: list, evaluation: list, predictions: pd.array, validation: pd.array):
        predictions_close = predictions[:, :, 0]
        validation_close = validation[:, :, 0]

        fig = plt.figure(figsize=(20, 10), layout="constrained")
        spec = fig.add_gridspec(3, 4)

        ax = fig.add_subplot(spec[0, :])
        fig.suptitle(self.name)
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

        ax = fig.add_subplot(spec[1, 3])
        ax.set_title("Symmetric Mean Absolute Percentage Error")
        ax.plot(results.epoch, results.history['metric_smape'], label=f'Train: {np.mean(results.history['metric_smape']):.4f}')
        ax.plot(results.epoch, results.history['val_metric_smape'], label=f'Validation: {np.mean(results.history['val_metric_smape']):.4f}')
        ax.legend()

        ax = fig.add_subplot(spec[2, 0])
        ax.set_title("Coefficient of Determination")
        ax.plot(results.epoch, results.history['metric_r2score'], label=f'Train: {np.mean(results.history['metric_r2score']):.4f}')
        ax.plot(results.epoch, results.history['val_metric_r2score'], label=f'Validation: {np.mean(results.history['val_metric_r2score']):.4f}')
        ax.legend()

        ax = fig.add_subplot(spec[2, 1])
        ax.set_title("Training data")
        ax.pie(sizes, labels=sizes)
        ax.legend()

        ax = fig.add_subplot(spec[2, 2])
        ax.set_title("Test evaluation")
        ax.bar(['a', 'b', 'c', 'd'], evaluation)
        for i, v in enumerate(evaluation):
            ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')
        plt.show()

    def load_model(self, path: str = config.models_path):
        self.model = keras.saving.load_model(f'{path}/{self.name}.keras')
        self.scaler = joblib.load(f'{path}/{self.name}.save')

    def save_model(self, path: str = config.models_path):
        if self.model != None:
            self.model.save(f'{path}/{self.name}.keras')
            joblib.dump(self.scaler, f'{path}/{self.name}.save')
