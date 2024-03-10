import ccxt
import numpy as np
import pandas as pd
import ta
import pandas_ta as pdta
from pandas import DataFrame
import finta
import math
import statistics
from binance.client import Client
import matplotlib.pyplot as plt
import pandas_ta
import yfinance as yf
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.utils import class_weight
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adam
from sklearn.calibration import CalibratedClassifierCV
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import LogCosh
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Assuming 'returns' are the actual returns


global_timeframe = "1H"
ratio = "T"
symbol = "ETHUSDT"
crypto =True

path = "/Users/davidhuber/Desktop/Financial_data/Final/{}_{}.csv".format(symbol, global_timeframe)

df = pd.read_csv(path)

def sharpe_ratio_loss_with_penalty(y_true, y_pred, risk_free_rate=0.0, non_trading_penalty=0.1, epsilon=1e-6):
    """
    Sharpe Ratio loss function with a penalty for non-trading predictions.
    Args:
    - y_true: Tensor of actual returns.
    - y_pred: Tensor of predicted returns.
    - risk_free_rate: Risk-free rate, default is 0.0.
    - non_trading_penalty: Penalty factor for non-trading predictions.
    - epsilon: Small value to prevent division by zero.

    Returns:
    - Loss value incorporating Sharpe Ratio and non-trading penalty.
    """
    # Calculate excess return over risk-free rate
    excess_return = y_pred - risk_free_rate

    # Mean of excess return
    mean_excess_return = tf.reduce_mean(excess_return)

    # Standard deviation of excess return
    std_excess_return = tf.math.reduce_std(excess_return) + epsilon

    # Sharpe Ratio
    sharpe_ratio = mean_excess_return / std_excess_return

    # Penalty for non-trading predictions (predictions close to zero)
    non_trading_penalty_term = non_trading_penalty * tf.reduce_mean(tf.square(y_pred))

    # Combine Sharpe Ratio and penalty term (minimize negative Sharpe Ratio and penalty)
    return -sharpe_ratio + non_trading_penalty_term


def weighted_binary_crossentropy(y_true, y_pred):
    # Assign higher weight to the minority class
    # Adjust these weights as needed
    class_weight_0 = tf.constant(0.15, dtype=tf.float32)  # for '0' class
    class_weight_1 = tf.constant(0.85, dtype=tf.float32)  # for '2' class (treated as '1')

    # Calculate the binary cross-entropy loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    standard_loss = bce(y_true, y_pred)

    # Convert y_true to float for the multiplication
    y_true_float = tf.cast(y_true, tf.float32)

    # Apply the weights
    weighted_loss = standard_loss * (y_true_float * class_weight_1 + (1 - y_true_float) * class_weight_0)

    return weighted_loss

class TradingLoss(tf.keras.losses.Loss):
    def __init__(self, beta=2.5, threshold=0.0, name="trading_loss"):
        super().__init__(name=name)
        self.beta = beta
        self.threshold = threshold

    def call(self, y_true, y_pred):
        # Separate labels and actual returns
        # Assuming labels are in the first column and returns in the second
        labels = y_true[:, 0]
        actual_returns = y_true[:, 1]

        # Reshape y_pred to match the shape of labels
        y_pred = tf.squeeze(y_pred)

        # Calculate binary cross-entropy
        bce = tf.keras.losses.binary_crossentropy(labels, y_pred, from_logits=True)

        # Adjust loss based on actual returns and decisions
        bad_decision_mask = tf.logical_or(
            tf.logical_and(tf.equal(labels, 1), tf.less_equal(actual_returns, self.threshold)),
            tf.logical_and(tf.equal(labels, 0), tf.greater(actual_returns, self.threshold))
        )

        adjusted_loss = bce * (self.beta * tf.cast(bad_decision_mask, tf.float32) + 1.0)

        return tf.reduce_mean(adjusted_loss)

class TradingLossWithRewards(tf.keras.losses.Loss):
    def __init__(self, beta=2.5, reward_factor=0.5, threshold=0.0, name="trading_loss"):
        super().__init__(name=name)
        self.beta = beta
        self.reward_factor = reward_factor
        self.threshold = threshold

    def call(self, y_true, y_pred):
        labels = y_true[:, 0]
        actual_returns = y_true[:, 1]
        binary_predictions = tf.cast(tf.greater_equal(y_pred, self.prediction_threshold), tf.float32)
        y_pred = tf.squeeze(y_pred)
        bce = tf.keras.losses.binary_crossentropy(labels, y_pred, from_logits=True)

        # Mask for bad decisions
        bad_decision_mask = tf.logical_or(
            tf.logical_and(tf.equal(labels, 1), tf.less_equal(actual_returns, self.threshold)),
            tf.logical_and(tf.equal(labels, 0), tf.greater(actual_returns, self.threshold))
        )

        # Mask for good decisions
        good_decision_mask = tf.logical_not(bad_decision_mask)

        # Adjust loss based on decisions
        #adjusted_loss = bce * (self.beta * tf.cast(bad_decision_mask, tf.float32) + 1.0) * (tf.exp((-actual_returns)/self.threshold)-1.0) * self.reward_factor
        adjusted_loss = bce * (self.beta * tf.cast(bad_decision_mask, tf.float32) + 1.0) * ((self.threshold - actual_returns)*10)

        # Reward for profitable trades
        #reward = self.reward_factor * tf.reduce_mean(tf.cast(good_decision_mask, tf.float32) * actual_returns*100)
        return_al = labels * (actual_returns - self.threshold) * 10
        reward = (return_al * self.reward_factor)
        # Combine loss and reward
        total_loss = adjusted_loss - reward

        return tf.reduce_mean(total_loss)

class SharpeRatioLossWithPenalty(tf.keras.losses.Loss):
    def __init__(self, risk_free_rate=0.0, non_trading_penalty=0.1, threshold=0.5, epsilon=1e-6, name="sharpe_ratio_loss"):
        super().__init__(name=name)
        self.risk_free_rate = risk_free_rate
        self.non_trading_penalty = non_trading_penalty
        self.threshold = threshold
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """
        Calculate the Sharpe Ratio-based loss with a penalty for not trading.
        Args:
        - y_true: 2D tensor with labels in the first column and actual returns in the second.
        - y_pred: Tensor of model predictions.

        Returns:
        - A tensor representing the loss.
        """
        # Separate labels and actual returns
        labels = y_true[:, 0]
        actual_returns = y_true[:, 1]

        # Convert probabilities to binary trading signals
        trading_signals = tf.cast(tf.greater(y_pred, self.threshold), tf.float32)

        # Calculate strategy returns based on signals
        strategy_returns = trading_signals * actual_returns

        # Compute Sharpe Ratio components
        excess_return = tf.reduce_mean(strategy_returns - self.risk_free_rate)
        volatility = tf.math.reduce_std(strategy_returns) + self.epsilon

        # Negative Sharpe Ratio as base loss
        sharpe_ratio_loss = -(excess_return / volatility)

        # Penalty for not trading (when signal is 0)
        non_trading_penalty_term = self.non_trading_penalty * tf.reduce_mean(1.0 - trading_signals)

        # Total loss with non-trading penalty
        total_loss = sharpe_ratio_loss + non_trading_penalty_term

        return total_loss

class ExponentialRiskAversionLoss(tf.keras.losses.Loss):
    def __init__(self, beta=2.5, threshold=0.0, risk_aversion=1.0, prediction_threshold=0.5, penalty=100, name="exp_risk_aversion_loss"):
        super().__init__(name=name)
        self.beta = beta
        self.threshold = threshold
        self.risk_aversion = risk_aversion
        self.prediction_threshold = prediction_threshold
        self.penalty = penalty

    def call(self, y_true, y_pred):
        labels = y_true[:, 0]

        actual_returns = y_true[:, 1]
        binary_predictions = tf.cast(tf.greater_equal(y_pred, self.prediction_threshold), tf.float32)
        binary_predictions_not = tf.cast(tf.less(y_pred, self.prediction_threshold), tf.float32)
        y_pred = tf.squeeze(y_pred)

        # Calculate binary cross-entropy
        bce = tf.keras.losses.binary_crossentropy(labels, y_pred, from_logits=True)

        # Adjust loss based on actual returns and decisions
        bad_decision_mask = tf.logical_or(
            tf.logical_and(tf.equal(labels, 1), tf.less_equal(actual_returns, self.threshold)),
            tf.logical_and(tf.equal(labels, 0), tf.greater(actual_returns, self.threshold))
        )

        # Incorporate risk aversion using an exponential function
        #risk_aversion_loss = tf.exp(-self.risk_aversion * binary_predictions *(actual_returns/self.threshold))-1
        #risk_aversion_loss = tf.exp(-self.risk_aversion * actual_returns)-1
        #risk_aversion_loss = binary_predictions*tf.exp(-self.risk_aversion * (actual_returns - self.threshold)) - 1
        #risk_aversion_loss = -tf.exp(1/self.risk_aversion * (actual_returns - self.threshold))+1
        #risk_aversion_loss = 1.2**(-self.risk_aversion * actual_returns/self.threshold)
        #risk_aversion_loss = binary_predictions * 1.5 ** (-self.risk_aversion * actual_returns / self.threshold) -1 -(binary_predictions_not * 1.2 ** (-self.risk_aversion * actual_returns / self.threshold) -1)
        #risk_aversion_loss = binary_predictions * 1.5 ** (-self.risk_aversion * actual_returns / self.threshold) -1
        #risk_aversion_loss = 1.2 ** (-self.risk_aversion * (binary_predictions * actual_returns + binary_predictions_not * actual_returns))
        #risk_aversion_loss = ((-actual_returns/self.threshold))**1
        #risk_aversion_loss = ((-actual_returns) * 1000) ** 3
        # Combine BCE loss, bad decision penalty, and risk aversion
        #total_loss = bce * ((self.beta * tf.cast(bad_decision_mask, tf.float32)) + 1.0) * risk_aversion_loss * (1-1/self.penalty) + binary_predictions_not*1/self.penalty
        #total_loss = bce * ((self.beta * tf.cast(bad_decision_mask, tf.float32)) + 1.0) + ((binary_predictions * -actual_returns*3) + 1) + ((binary_predictions_not * actual_returns*1) + 1)
        #total_loss = (((- actual_returns)/self.threshold)**3) * bce * -binary_predictions_not + (((- actual_returns)/self.threshold)**3) * bce * binary_predictions * 2
        #total_loss = bce + binary_predictions * -(actual_returns/self.threshold) **3
        #total_loss = bce + y_pred * -((actual_returns+self.threshold)/self.threshold) **3 + (1-y_pred) * ((actual_returns+self.threshold)/self.threshold) **3  #+((1-y_pred) * actual_returns*100 - y_pred*actual_returns*100)- tf.math.reduce_mean(y_pred) #+ 100 * (1-y_pred))
        #total_loss = bce + 2* (0.7 - y_pred) * ((actual_returns + self.threshold) / self.threshold) ** 3  # +((1-y_pred) * actual_returns*100 - y_pred*actual_returns*100)- tf.math.reduce_mean(y_pred) #+ 100 * (1-y_pred))
        total_loss = (y_pred/0.5) * ((-actual_returns)/self.threshold)**1# + (1-y_pred) * ((actual_returns+self.threshold)/self.threshold) **3  # +((1-y_pred) * actual_returns*100 - y_pred*actual_returns*100)- tf.math.reduce_mean(y_pred) #+ 100 * (1-y_pred))
        market_loss = 1 * ((-actual_returns)/self.threshold)**1
        total_loss = total_loss + (y_pred/0.5)*0.05/100
        excess_ret = total_loss - market_loss
        diff_SR = (tf.math.reduce_mean(total_loss))/(tf.math.reduce_std(total_loss))-(tf.math.reduce_mean(market_loss))/(tf.math.reduce_std(market_loss))
         #* ((actual_returns + self.threshold) / self.threshold) ** 3  # +((1-y_pred) * actual_returns*100 - y_pred*actual_returns*100)- tf.math.reduce_mean(y_pred) #+ 100 * (1-y_pred))

        #total_loss = bce * (1 - y_pred)
        #SR = tf.math.reduce_mean(binary_predictions *((actual_returns/self.threshold) **1))/ tf.math.reduce_min(binary_predictions *((actual_returns/self.threshold) **1))

        return 6*diff_SR + 3*tf.math.reduce_mean(excess_ret) + 5*total_loss/(tf.math.reduce_sum(tf.math.maximum(total_loss, 0.0)+1)) #+ tf.exp(((tf.math.reduce_sum(binary_predictions)/(tf.math.reduce_sum(binary_predictions)+tf.math.reduce_sum(binary_predictions_not))))/0.10) #+ 3*(tf.math.reduce_mean(total_loss))/(tf.math.reduce_std(total_loss)) #+ abs((0.15 - (tf.math.reduce_sum(binary_predictions)/(tf.math.reduce_sum(binary_predictions)+tf.math.reduce_sum(binary_predictions_not))))*3)**2

class SharpeRatioTradingLoss(tf.keras.losses.Loss):
    def __init__(self, risk_free_rate=0.0, epsilon=1e-6, prediction_threshold=0.5, name="sharpe_ratio_trading_loss"):
        super().__init__(name=name)
        self.risk_free_rate = risk_free_rate
        self.epsilon = epsilon
        self.prediction_threshold = prediction_threshold

    def call(self, y_true, y_pred):
        labels = y_true[:, 0]
        actual_returns = y_true[:, 1]

        # Convert predictions to binary trading decisions
        trading_decisions = tf.cast(tf.greater(y_pred, self.prediction_threshold), tf.float32)

        # Compute strategy returns based on trading decisions
        strategy_returns = trading_decisions * actual_returns

        # Compute Sharpe Ratio components
        excess_return = tf.reduce_mean(strategy_returns - self.risk_free_rate)
        volatility = tf.math.sqrt(tf.math.reduce_variance(strategy_returns) + self.epsilon)

        # Negative Sharpe Ratio as part of the loss
        sharpe_ratio_loss = -(excess_return / volatility)

        # Calculate binary cross-entropy loss
        bce = tf.keras.losses.binary_crossentropy(labels, y_pred, from_logits=False)

        # Combine BCE loss and Sharpe Ratio loss
        combined_loss = bce + sharpe_ratio_loss

        return combined_loss


# Initialize EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=15,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)


if crypto == False :
    df['Gmt time'] = df['Gmt time'].str.replace(".000","")
    df['Gmt time'] = pd.to_datetime(df['Gmt time'],format='%d.%m.%Y %H:%M:%S')
    df['timestamp'] = df['Gmt time']
    df['date'] = df['timestamp']
    #df = df.drop(['Gmt time'], axis=1)
    #df.index = df['date']
    df['close'] = pd.to_numeric(df['Close'])
    df['high'] = pd.to_numeric(df['High'])
    df['low'] = pd.to_numeric(df['Low'])
    df['open'] = pd.to_numeric(df['Open'])
    df['volume'] = pd.to_numeric(df['Volume'])
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], unit='ms')
    df['hour'] = df['Gmt time'].dt.hour
    df['minute'] = df['Gmt time'].dt.minute
    df['hour+minute'] = df['hour'] + df['minute'] / 100
    df = df.drop(['Gmt time', 'Close', 'Low', 'Open', 'High'], axis=1)
    df = df.drop(['Gmt time.1'], axis=1)

elif crypto == True :
    df['date'] = pd.to_datetime(df['timestamp'])
    #df.index = df['date']
    df['timestamp'] = pd.to_datetime(df['timestamp'])#, unit='ms')
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['hour+minute'] = df['hour'] + df['minute'] / 100

data = df

print(df.head())

#technicals :

data['ema_200'] = ta.wrapper.EMAIndicator(data['close'], 200, fillna= True).ema_indicator()/ data.close

data['ema_100'] = ta.wrapper.EMAIndicator(data['close'], 100, fillna= True).ema_indicator()/ data.close
data['ema_50'] = ta.wrapper.EMAIndicator(data['close'], 50, fillna= True).ema_indicator()/ data.close
data['ema_20'] = ta.wrapper.EMAIndicator(data['close'], 20, fillna= True).ema_indicator()/ data.close
data['ema_10'] = ta.wrapper.EMAIndicator(data['close'], 10, fillna= True).ema_indicator()/ data.close


data['ema_200_slope'] = (data.ema_200 - data.ema_200.shift(30)) / data.ema_200.shift(30)

data['ema_100_slope'] = (data.ema_100 - data.ema_100.shift(20)) / data.ema_100.shift(20)
data['ema_50_slope'] = (data.ema_50 - data.ema_50.shift(10)) / data.ema_50.shift(10)
data['ema_20_slope'] = (data.ema_20 - data.ema_20.shift(5)) / data.ema_20.shift(5)
data['ema_10_slope'] = (data.ema_10 - data.ema_10.shift(3)) / data.ema_10.shift(3)

data['macd'] = ta.wrapper.MACD(data['close'], 26, 12, 9, True).macd()
data['macd_diff'] = ta.wrapper.MACD(data['close'], 26, 12, 9, True).macd_diff()
data['macd_sign'] = ta.wrapper.MACD(data['close'], 26, 12, 9, True).macd_signal()

data['rsi'] = ta.wrapper.RSIIndicator(data['close'], 14, fillna=True).rsi()
data['rsi_2'] = ta.wrapper.RSIIndicator(data['close'], 28, fillna=True).rsi()

data['rsi_s'] = data['rsi'].shift(2)
data['rsi_2_s'] = data['rsi_2'].shift(2)

data['rsi_slope'] = data['rsi'] / data['rsi_s']
data['rsi_2_slope'] = data['rsi_2'] / data['rsi_2_s']

data['stoch_rsi_k'] = ta.wrapper.StochRSIIndicator(data['close'], 14, 3, 3, True).stochrsi_k()
data['stoch_rsi_d'] = ta.wrapper.StochRSIIndicator(data['close'], 14, 3, 3, True).stochrsi_d()

data['stoch_rsi_k1'] = ta.wrapper.StochRSIIndicator(data['close'], 9, 3, 3, True).stochrsi_k()
data['stoch_rsi_d1'] = ta.wrapper.StochRSIIndicator(data['close'], 9, 3, 3, True).stochrsi_d()

data['stoch_rsi_k_s'] = data['stoch_rsi_k'].shift(2)
data['stoch_rsi_d_s'] = data['stoch_rsi_d'].shift(2)

data['stoch_rsi_k1_s'] = data['stoch_rsi_k1'].shift(2)
data['stoch_rsi_d1_s'] = data['stoch_rsi_d1'].shift(2)


data['EOM'] = ta.wrapper.EaseOfMovementIndicator(data.high,data.low,data.volume, 14, fillna=True).ease_of_movement()

data['adx'] = ta.wrapper.ADXIndicator(data.high, data.low, data.close, 14,True).adx()
data['adx_c'] = data.adx.shift(3)-data.adx / data.adx

data['adxp'] = ta.wrapper.ADXIndicator(data.high, data.low, data.close, 14,True).adx_pos()
data['adx_p_c'] = data.adxp.shift(3)-data.adxp / data.adxp

data['adxn'] = ta.wrapper.ADXIndicator(data.high, data.low, data.close, 14,True).adx_neg()
data['adx_n_c'] = data.adxn.shift(3)-data.adxn / data.adxn

data['std'] = pandas_ta.stdev(data.close, 5) / data.close
data['std2'] = pandas_ta.stdev(data.close, 50)/ data.close
data['std3'] = pandas_ta.stdev(data.close, 200)/ data.close
data['stc'] = ta.wrapper.STCIndicator(data.close,50,23,10,3,3,True).stc()


data['min'] = data.low.rolling(20).min() / data.close
data['max'] = data.high.rolling(20).max() / data.close

data['min1'] = data.low.rolling(50).min() / data.close
data['max1'] = data.high.rolling(50).max() / data.close

data['min2'] = data.low.rolling(200).min() / data.close
data['max2'] = data.high.rolling(200).max() / data.close

data['Vwap'] = ta.wrapper.VolumeWeightedAveragePrice(data.high, data.low, data.close, data.volume, 14, True).vwap / data.close
data['Vwap2'] = ta.wrapper.VolumeWeightedAveragePrice(data.high, data.low, data.close, data.volume, 50, True).vwap / data.close
data['Vwap3'] = ta.wrapper.VolumeWeightedAveragePrice(data.high, data.low, data.close, data.volume, 5, True).vwap / data.close



data['returns'] =(data['close'].shift(-5) - data['close'])/data['close']
data['returns1'] =(data['close'].shift(-4) - data['close'])/data['close']
data['returns2'] =(data['close'].shift(-3) - data['close'])/data['close']
data['returns3'] =(data['close'].shift(-2) - data['close'])/data['close']
data['returns4'] =(data['close'].shift(-1) - data['close'])/data['close']

# Apply the custom function to each row in the DataFrame

data['last_return'] = data['returns'].shift(5)
data['last_return1'] = data['returns1'].shift(4)
data['last_return2'] = data['returns2'].shift(3)
data['last_return3'] = data['returns3'].shift(2)
data['last_return4'] = data['returns4'].shift(1)

data['volatility'] = data['returns4'].rolling(window=7).std()


value = 0
threshold = 0.0001

for index,row in data.iterrows():
    if row['returns4'] >= threshold:
        value = 1
    elif row['returns4'] <= -threshold:
        value = -1
    else:
        value = 0

    data.at[index, 'returns_b'] = value

data['pos_ret'] = np.where(data['returns_b']> 0, 1, 0)
data['neg_ret'] = np.where(data['returns_b']< 0, 1, 0)

data['pos_ret_s']=data['pos_ret'].shift(1)

prob_long = (len(data['pos_ret'])-sum(data['pos_ret']))/len(data['pos_ret'])
prob_short = (len(data['neg_ret'])-sum(data['neg_ret']))/len(data['neg_ret'])

print(prob_long)
print(prob_short)



data = data.dropna()


half_df = round(len(data)/1.5)
data_test = data.iloc[:half_df]
data_run = data.iloc[half_df:]



data = data_test


X = data.drop(['volatility', 'returns', 'returns1', 'returns2', 'returns3', 'returns4', 'date', 'timestamp', 'returns_b', 'neg_ret','pos_ret', 'high', 'low', 'close', 'open'], axis=1)
labels = data['pos_ret'].values



# Function to create sequences
def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X.iloc[i: i + sequence_length])  # Sequence of feature rows
        y_seq.append(y[i + sequence_length - 1])  # Corresponding label
    return np.array(X_seq), np.array(y_seq)

# Assuming 'data' is your DataFrame
X = data.drop(['volatility', 'returns', 'returns1', 'returns2', 'returns3', 'returns4', 'date', 'timestamp', 'returns_b', 'neg_ret', 'pos_ret', 'high', 'low', 'close', 'open'], axis=1)
labels = data['pos_ret'].values
returns = data['returns4'].values

# Combine labels and returns
y_train_combined = np.column_stack((labels, returns))
y = y_train_combined


# Define the sequence length (number of time steps)
sequence_length = 3

# Create sequences
X_seq, y_seq = create_sequences(X, y, sequence_length=sequence_length)

# Assuming 'X_seq' is a list of sequences and 'y_seq' is the corresponding labels
# Flatten all sequences into a single 2D array for scaling
all_sequences_flat = np.concatenate(X_seq, axis=0)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the flattened data
scaler.fit(all_sequences_flat)

# Transform each sequence
X_seq_scaled = np.array([scaler.transform(seq) for seq in X_seq])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_seq_scaled, y_seq, test_size=0.05, random_state=42)

# Reshape for LSTM input
sequence_length = X_seq[0].shape[0]  # Assuming all sequences have the same length
X_train_reshaped = X_train.reshape((X_train.shape[0], sequence_length, -1))
X_test_reshaped = X_test.reshape((X_test.shape[0], sequence_length, -1))




# Initialize and train the LSTM model
model = Sequential()
model.add(LSTM(70, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.3))
"""model.add(LSTM(10, return_sequences=False))
model.add(Dropout(0.3))"""
"""model.add(LSTM(50, return_sequences=False))  # Set return_sequences to False here
model.add(Dropout(0.2))"""
model.add(Dense(1, activation='sigmoid'))
# Compile the model (use your preferred loss function and optimizer)
model.compile(optimizer=Adam(learning_rate=0.0008), loss=ExponentialRiskAversionLoss(beta=3, threshold=threshold, risk_aversion=0.5, prediction_threshold=0.5), metrics=['accuracy'])
#model.compile(optimizer=Nadam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels.flatten())
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Fit the model with class weights
model.fit(X_train_reshaped, y_train, epochs=200, batch_size=500, validation_data=(X_test_reshaped, y_test), callbacks=[early_stopping], class_weight=class_weights_dict) #callbacks=[early_stopping]

data = data_run


def create_sequences_predict(X, sequence_length):
    X_seq = []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X.iloc[i: i + sequence_length])
    return np.array(X_seq)

# Assuming 'data' is your DataFrame
# The same sequence length used during training

# Prepare X with the same features used during training
X = data.drop(['volatility', 'returns', 'returns1', 'returns2', 'returns3', 'returns4', 'date', 'timestamp', 'returns_b', 'neg_ret', 'pos_ret', 'high', 'low', 'close', 'open'], axis=1)

# Create sequences from X
X_sequences = create_sequences_predict(X, sequence_length)

# Scale the sequences
X_sequences_scaled = np.array([scaler.transform(seq) for seq in X_sequences])

# Reshape for LSTM input
X_sequences_scaled_reshaped = X_sequences_scaled.reshape((X_sequences_scaled.shape[0], sequence_length, X.shape[1]))

# Use the model to predict
predictions_proba1 = model.predict(X_sequences_scaled_reshaped)

# Reshape predictions from 3D to 1D
predictions_flat = predictions_proba1.reshape(-1)

type_plot = 0
leverage = 1
prob = 0.5
fee_amount = 0.05/100
while type_plot<2 :
    # Flatten the predictions and assign them to the DataFrame

    # Now you can safely assign it to your DataFrame
    # Initialize a column with NaNs
    data['Predictions3'] = np.nan

    start_idx = sequence_length - 1

    # Assign predictions to the DataFrame
    data['Predictions3'].iloc[start_idx:] = np.where(predictions_flat > prob, 1 * leverage, 0)

    # Initialize a column for fees
    data['Fees'] = 0.0

    # Calculate fees based on signal changes
    for i in range(start_idx + 1, len(data)):
        if data['Predictions3'].iloc[i] != data['Predictions3'].iloc[i - 1]:
            # Subtract fee when the signal changes
            data['Fees'].iloc[i] = fee_amount


    # Subtract fees from Strategy Returns
    data['StrategyReturns_LSTM_long'] = data['Predictions3'] * data['returns4'] - data['Fees']*leverage

    #data['Predictions3'].iloc[start_idx:] = np.where(predictions_flat > prob, -(predictions_flat*leverage/prob), 0)

    print(data.head(50))

    # Calculate the equity curve
    data['EquityCurve_LSTM_long'] =np.log10((1 + data['StrategyReturns_LSTM_long']).cumprod())

    data.index = data['timestamp']
    # Plot the equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(data['EquityCurve_LSTM_long'], label='LSTM long', color= "green")
    plt.plot(np.log10((1+data['returns4']).cumprod()), label = 'stock', color= "black")
    plt.title('Backtest Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.show()
    type_plot = float(input("continue plotting = (<2 == continue) "))
    fee_amount = float(input("fee = "))/100
    leverage = float(input("leverage = "))
    prob = float(input("prob = "))
    save = float(input("prob = "))
