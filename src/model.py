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
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import keras_tuner

# Algorithm for calculating Symmetric Mean Absolute Percentage Error
@keras.saving.register_keras_serializable()
def metric_aggregated(y_true, y_pred):
    return (
        metric_rmse(y_true, y_pred) -
        metric_r2score(y_true, y_pred))

@keras.saving.register_keras_serializable()
def metric_smape(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    epsilon = 1e-10 
    diff = tf.abs(y_true - y_pred)
    scale = tf.abs(y_true) + tf.abs(y_pred) + epsilon
    return 200 * tf.reduce_mean(diff / scale)

@keras.saving.register_keras_serializable()
def metric_rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Algorithm for calculating Coefficient of Determination
@keras.saving.register_keras_serializable()
def metric_r2score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

class StockModel:

    def __init__(self, frame: pd.DataFrame, name: str, force_tuner=False):
        # ====================================================
        # Configuration
        # ====================================================
        self.train_size = 0.7   # The size of the training data as percentage of total datasize
        self.valid_size = 0.15  # The size of the validation data as percentage of total datasize
        self.steps = 10         # The historically observed datapoints (days)
        self.batchsize = 24
        self.patience = 6
        self.epochs = 100       # The maximum number of epocs
        self.name = name        # The unique name of the symbol
        self.force_tuner = force_tuner
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
            'SMA_low', 'SMA_high',
            'EMA_low', 'EMA_high',
            'DEMA_val', 'ROCR_val',
            'RSI_val', 'HURST_val'
            ]]
        
        self.features = len(self.dataset.columns)
        
    def train_model(self, interactive: bool = True) -> Self:   
        # ====================================================
        # Fit the transform to training data
        # ====================================================
        scaled_dataset = self.scaler.fit_transform(
            self.dataset)
        
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

        # ====================================================
        # Create timeseries
        # ====================================================
        x_train, y_train = self.create_sequences(train_set)
        x_valid, y_valid = self.create_sequences(valid_set)
        x_tests, y_tests = self.create_sequences(tests_set)
        
        # ====================================================
        # Create LSTM pipeline
        # ====================================================
        def build_model(hp):
            units = hp.Int('units',  min_value=48, max_value=112, step=16)
            learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
            
            model = Sequential([
                Input(shape=(self.steps, self.features)),
                LSTM(units, activation='relu', return_sequences=True),
                LSTM(units),
                Dense(self.features, activation="linear")
            ])
            # ====================================================
            # Compile model
            # ====================================================
            model.compile( 
                optimizer=Adam(learning_rate=learning_rate),
                loss=metric_aggregated,
                metrics=[
                    'mse',
                    'mae',
                    metric_smape,
                    metric_r2score,
                    metric_rmse
                ])
            return model
                
        # ====================================================
        # Create tuner
        # ====================================================
        tuner = keras_tuner.GridSearch(
            build_model,
            tune_new_entries=True,
            objective='val_loss',
            overwrite=self.force_tuner,
            directory=config.tuner_path,
            project_name=self.name
        )
        # ====================================================
        # Create stopping function to avoid overfitting
        # ====================================================
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience, 
            restore_best_weights=True)

        # ====================================================
        # Perform search
        # ====================================================
        tuner.search(x_train, y_train, 
            epochs=15, 
            validation_data=(x_valid, y_valid), 
            batch_size=self.batchsize)
        
        # Retrain best model
        best_hp = tuner.get_best_hyperparameters()[0]
        self.model = tuner.hypermodel.build(best_hp)

        # ====================================================
        # Train the model
        # ====================================================
        result = self.model.fit(
            x_train,
            y_train,
            batch_size=self.batchsize,
            epochs=self.epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # ====================================================
        # Evaluate model
        # ====================================================
        eval_metrics = self.model.evaluate(x_tests, y_tests)
        
        # ====================================================
        # Debug print - Training result
        # ====================================================        
        self.print_metrics(eval_metrics, best_hp.values)

        # ====================================================
        # Make predictions on validation data
        # ====================================================    
        predi_set = self.model.predict(x_tests)

        # ====================================================
        # Debug print
        # ====================================================        
        self.print_frame(self.dataset, 'Original')
        self.print_frame(self.convert_to_frame(scaled_dataset), 'Scaled')
        self.print_frame(self.convert_to_frame(predi_set), 'Predictions')

        # ====================================================
        # Plot metrics
        # ====================================================
        self.plot_metrics(
            result,
            interactive,
            eval_metrics,
            (
                train_set,
                valid_set,
                tests_set,
                predi_set,
            )
        )
        return self

    def create_sequences(self, data: np.array) -> tuple[np.array, np.array]:
        X, y = [], []

        for i in range(len(data) - self.steps):
            X.append(data[i:i + self.steps, :])
            y.append(data[i + self.steps, :])
        return np.array(X), np.array(y)

    def inverse_transform(self, data: np.array) -> pd.DataFrame:
        # ===========================================
        # Reshapes from timeseries to original dataframe
        # ===========================================
        # (samples, horizon, features) => (samples, features)
        return self.convert_to_frame(
            self.scaler.inverse_transform(data))

    def convert_to_frame(self, data: np.array) -> pd.DataFrame:
        return pd.DataFrame(data, columns=self.dataset.columns)

    def print_frame(self, data: pd.DataFrame, name: str):
        print('\n=========================================')
        print(f'Dataset: {name}')
        print('=========================================\n')
        print(data.tail(10))

    def print_metrics(self, data: list, hp: dict):
        print('\n=========================================')
        print('Hyperparameters')
        print('=========================================\n')
        print(f'Units:\t{hp['units']}')
        print(f'Rate:\t{hp['learning_rate']}')

        print('\n=========================================')
        print('Metrics')
        print('=========================================\n')
        print(f'Loss:\t{data[0]}')
        print(f'MSE:\t{data[1]}')
        print(f'MAE:\t{data[2]}')
        print(f'SMAPE:\t{data[3]}')
        print(f'R2Score:\t{data[4]}')
        print(f'RMSE:\t{data[4]}')


    def print_predictions(self, data: np.array):
        print(f'\nClosing values:\n\n{
            self.dataset.tail(1)
        }')

        print(f'\nPredicted values (By day):\n\n{
            data
        }')

    def predict(self, days: int = 30) -> pd.DataFrame:
        # ====================================================
        # Scale data
        # ====================================================
        scaled_dataset = self.scaler.fit_transform(self.dataset)
    
        predictions = scaled_dataset[-self.steps:]

        for _ in range(days):
            x = predictions[-self.steps:]
            x = x.reshape((1, self.steps, self.features))
            out = self.model.predict(x)
            predictions = np.append(predictions, out, axis=0)
        predictions = predictions[self.steps-1:]

        # ====================================================
        # Reshape data for presentation
        # ====================================================
        # Input: Timeseries shape (samples, horizon, features)
        # Output: (samples, features) using the first horizon of every sample
        rescaled_predictions = self.inverse_transform(
            predictions)

        # ===========================================
        # Create a date range in the future
        # =========================================== 
        # This way a dataframe is built matching
        # the original dataset closely.
        # Using the last day in the dataset +1 to be
        # the first day of prediction
        index = pd.date_range(self.dataset.index[-1], periods=rescaled_predictions.shape[0]+1)[1:]
        # Apply index 
        rescaled_predictions = rescaled_predictions.set_index(index)
        # ====================================================
        # Print prognosis
        # ====================================================
        self.print_predictions(
            rescaled_predictions
        )
        return rescaled_predictions
    
    def plot_metrics(self, results: any, interactive: bool, evaluation: list, data: tuple[np.array, np.array, np.array, np.array]):
        (train, valid, tests, predi) = data

        self.fig = plt.figure(figsize=(20, 10), layout="constrained")
        spec = self.fig.add_gridspec(3, 4)

        ax = self.fig.add_subplot(spec[0, :])
        self.fig.suptitle(self.name)
        ax.plot(tests[:,0], label="Test Values")
        ax.plot(predi[:,0], label="Predictions")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(True)
        ax.legend()

        ax = self.fig.add_subplot(spec[1, 0])
        ax.set_title("Loss (MSE)")
        ax.plot(results.epoch, results.history['loss'], label=f'Train: {np.mean(results.history['loss']):.4f}')
        ax.plot(results.epoch, results.history['val_loss'], label=f'Validation: {np.mean(results.history['val_loss']):.4f}')
        ax.legend()

        ax = self.fig.add_subplot(spec[1, 1])
        ax.set_title("Root Mean Square Error")
        ax.plot(results.epoch, tf.sqrt(results.history['metric_rmse']), label=f'Train: {np.mean(tf.sqrt(results.history['metric_rmse'])):.4f}')
        ax.plot(results.epoch, tf.sqrt(results.history['val_metric_rmse']), label=f'Validation: {np.mean(tf.sqrt(results.history['val_metric_rmse'])):.4f}')
        ax.legend()

        ax = self.fig.add_subplot(spec[1, 2])
        ax.set_title("Mean Absolut Error")
        ax.plot(results.epoch, results.history['mae'], label=f'Train: {np.mean(results.history['mae']):.4f}')
        ax.plot(results.epoch, results.history['val_mae'], label=f'Validation: {np.mean(results.history['val_mae']):.4f}')
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
        ax.pie([len(train), len(valid), len(tests)] , labels=[f'Train: {len(train)}', f'Valid: {len(valid)}', f'Test: {len(tests)}'])

        ax = self.fig.add_subplot(spec[2, 2])
        ax.set_title("Test evaluation")
        ax.axis("off")
        ax.table(rowLabels=['Loss', 'MSE', 'MAE', 'SMAPE', 'R2Score', 'RMSE'], cellText=[['%.4f' % x] for x in evaluation], loc='center')
        
        if interactive:
            plt.show(block=True)

    def load_model(self, path: str = config.model_path) -> Self:
        self.model = keras.saving.load_model(f'{path}/{self.name}.keras')
        self.scaler = joblib.load(f'{path}/{self.name}.save')
        return self

    def save_model(self, path: str = config.model_path) -> Self:
        if self.model != None:
            self.model.save(f'{path}/{self.name}.keras')
            joblib.dump(self.scaler, f'{path}/{self.name}.save')
        if self.fig != None:
            self.fig.savefig(f'{path}/{self.name}.png')
        return self

