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

    def __init__(self, frame: pd.DataFrame, name: str):
        # ====================================================
        # Configuration
        # ====================================================
        self.train_size = 0.7   # The size of the training data in percentage
        self.valid_size = 0.15  # The size of the validation data in percentage
        self.steps = 30         # The historically observed datapoints (days)
        self.horizon = 10        # The number of future datapoints to predict
        self.learning_rate = 0.0001 # The learning rate of the model
        self.epocs = 100        # The maximum number of epocs
        self.batchsize = self.steps
        self.units = 50
        self.patience = 6
        self.name = name        # The unique name of the symbol

        # ====================================================
        # Dashboard
        # ====================================================
        # The matplotlib dashboard of training metrics
        self.fig = None

        # ====================================================
        # The trained model
        # ====================================================
        # Either trained on the fly or loaded from disk
        self.model: Sequential = None

        # ====================================================
        # The scaler to be used
        # ====================================================
        # NOTE: The scaler configuration is serialized
        # together with the model as it needs to be aligned
        # with the training data. If the model is loaded from
        # file, the scaler needs to be restored separately.
        self.scaler = MinMaxScaler(
            feature_range=(0, 1))
        
        # ====================================================
        # Extract the features to train the model with
        # ====================================================
        # The first column which represents the Closing price
        # is to be considered the conclusive output for the
        # prediction (The "label").
        self.dataset = frame[[
            'Close',
            'Open', 'Volume',
            'High', 'Low',
            'SMA_high', 'SMA_low',
            'EMA_high', 'EMA_low',
            'DEMA_val', 'ROCR_val',
            'RSI_val'
            ]]
        
        self.features = len(self.dataset.columns)
        
    def train_model(self, interactive: bool = True) -> Self:
        '''
        Example input dataset. Shape: (Samples, Features):
        
        print(self.dataset.tail(3))

        Date                                                                   ...                                                         
        2024-11-07  133.070007  133.360001  3906400.0  134.800003  132.479996  ...  130.903124  130.035531  129.150186  1.035806  46.673107
        2024-11-08  134.339996  133.449997  3465700.0  135.020004  133.199997  ...  131.088901  130.697756  130.323147  1.055966  49.220698
        2024-11-11  133.000000  134.850006  2416862.0  135.222504  132.789993  ...  131.192204  131.051947  130.942071  1.056814  52.095027        
        '''

        # ====================================================
        # Fit the transform to training data
        # ====================================================
        scaled_dataset = pd.DataFrame(self.scaler.fit_transform(
            self.dataset), columns=self.dataset.columns)

        '''
        Example scaled data:

        print(scaled_dataset.tail(3))

                 Close      Open    Volume      High       Low  SMA_high   SMA_low  EMA_high   EMA_low  DEMA_val  ROCR_val   RSI_val
        3737  0.470015  0.474406  0.052429  0.474656  0.479428  0.611327  0.794623  0.767184  0.738193  0.718186  0.483080  0.468294
        3738  0.478107  0.474977  0.045042  0.476059  0.484025  0.614133  0.794876  0.768803  0.743686  0.727625  0.535304  0.495239
        3739  0.469568  0.483866  0.027461  0.477350  0.481407  0.616873  0.794577  0.769702  0.746624  0.732606  0.537500  0.525640
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
        
        '''
        Example of timeseries data shapes:

        print(x_train.shape)

        (2579, 30, 12)
        
        print(y_train.shape)
        (2579, 10, 12)

        '''
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
                
        # ====================================================
        # Make predictions on validation data
        # ====================================================
        predictions = self.model.predict(x_valid)

        # ====================================================
        # Reshape data for presentation
        # ====================================================
        # Input: Timeseries shape (samples, horizon, features)
        # Output: (samples, features) using the first horizon of every sample
        rescaled_predictions = self.inverse_transform(
            predictions[:,0:1].reshape(predictions.shape[0], predictions.shape[2]))
        
        rescaled_validations = self.inverse_transform(
            y_valid[:,0:1].reshape(y_valid.shape[0], y_valid.shape[2]))

        # ====================================================
        # Plot metrics
        # ====================================================
        self.plot_metrics(
            result,
            interactive,
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

    def inverse_transform(self, data: np.array) -> pd.DataFrame:
        # ===========================================
        # Reshapes from timeseries to original dataframe
        # ===========================================
        # (samples, horizon, features) => (samples, features)
        return pd.DataFrame(
            self.scaler.inverse_transform(data),
                columns=self.dataset.columns)

    def predict(self) -> pd.DataFrame:
        # ===========================================
        # Scale to fit 0 - 1
        # ===========================================
        # Extract the last N days from the most recent dataset
        scaled_dataset = pd.DataFrame(self.scaler.transform(
            self.dataset[-(self.steps*3):]), columns=self.dataset.columns)

        # ===========================================
        # Reshape scaled data
        # ===========================================
        # Shape needed is: (Samples, steps, features)
        x_predict, _ = self.create_sequences(scaled_dataset.values)
        
        # ===========================================
        # Perform prediction
        # ===========================================
        predictions = self.model.predict(x_predict)

        # ====================================================
        # Reshape data for presentation
        # ====================================================
        # Input: Timeseries shape (samples, horizon, features)
        # Output: (samples, features) using the first horizon of every sample
        rescaled_predictions = self.inverse_transform(
            predictions[:,0:1].reshape(predictions.shape[0], predictions.shape[2]))

        # ===========================================
        # Create a date range in the future
        # =========================================== 
        # This way a dataframe is built matching
        # the original dataset closely.
        # Using the last day in the dataset +1 to be
        # the first day of prediction
        index = pd.date_range(self.dataset.index[-1], periods=rescaled_predictions.shape[0]+1)[1:]
        # Apply index 
        return rescaled_predictions.set_index(index)
    
    def plot_metrics(self, results: any, interactive: bool, sizes: list, evaluation: list, predictions: pd.array, validation: pd.array):
        predictions_close = predictions['Close']
        validation_close = validation['Close']

        self.fig = plt.figure(figsize=(20, 10), layout="constrained")
        spec = self.fig.add_gridspec(3, 4)

        ax = self.fig.add_subplot(spec[0, :])
        self.fig.suptitle(self.name)
        ax.plot(validation_close, label="True Values")
        ax.plot(predictions_close, label="Predictions")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(True)
        ax.legend()

        ax = self.fig.add_subplot(spec[1, 0])
        ax.set_title("Loss (MAE)")
        ax.plot(results.epoch, results.history['loss'], label=f'Train: {np.mean(results.history['loss']):.4f}')
        ax.plot(results.epoch, results.history['val_loss'], label=f'Validation: {np.mean(results.history['val_loss']):.4f}')
        ax.legend()

        ax = self.fig.add_subplot(spec[1, 1])
        ax.set_title("Root Mean Square Error")
        ax.plot(results.epoch, tf.sqrt(results.history['mse']), label=f'Train: {np.mean(tf.sqrt(results.history['mse'])):.4f}')
        ax.plot(results.epoch, tf.sqrt(results.history['val_mse']), label=f'Validation: {np.mean(tf.sqrt(results.history['val_mse'])):.4f}')
        ax.legend()

        ax = self.fig.add_subplot(spec[1, 2])
        ax.set_title("Mean Square Error")
        ax.plot(results.epoch, results.history['mse'], label=f'Train: {np.mean(results.history['mse']):.4f}')
        ax.plot(results.epoch, results.history['val_mse'], label=f'Validation: {np.mean(results.history['val_mse']):.4f}')
        ax.legend()

        ax = self.fig.add_subplot(spec[1, 3])
        ax.set_title("Symmetric Mean Absolute Percentage Error")
        ax.plot(results.epoch, results.history['metric_smape'], label=f'Train: {np.mean(results.history['metric_smape']):.4f}')
        ax.plot(results.epoch, results.history['val_metric_smape'], label=f'Validation: {np.mean(results.history['val_metric_smape']):.4f}')
        ax.legend()

        ax = self.fig.add_subplot(spec[2, 0])
        ax.set_title("Coefficient of Determination")
        ax.plot(results.epoch, results.history['metric_r2score'], label=f'Train: {np.mean(results.history['metric_r2score']):.4f}')
        ax.plot(results.epoch, results.history['val_metric_r2score'], label=f'Validation: {np.mean(results.history['val_metric_r2score']):.4f}')
        ax.legend()

        ax = self.fig.add_subplot(spec[2, 1])
        ax.set_title("Training data")
        ax.pie(sizes, labels=sizes)
        ax.legend()

        ax = self.fig.add_subplot(spec[2, 2])
        ax.set_title("Test evaluation")
        ax.bar(['a', 'b', 'c', 'd'], evaluation)
        for i, v in enumerate(evaluation):
            ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')

        if interactive:
            plt.show(block=True)

    def load_model(self, path: str = config.models_path) -> Self:
        self.model = keras.saving.load_model(f'{path}/{self.name}.keras')
        self.scaler = joblib.load(f'{path}/{self.name}.save')
        return self

    def save_model(self, path: str = config.models_path) -> Self:
        if self.model != None:
            self.model.save(f'{path}/{self.name}.keras')
            joblib.dump(self.scaler, f'{path}/{self.name}.save')
        if self.fig != None:
            self.fig.savefig(f'{path}/{self.name}.png')
        return self

