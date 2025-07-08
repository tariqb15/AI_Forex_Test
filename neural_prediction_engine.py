# 🧠 BLOB AI - Neural Prediction Engine
# Advanced AI-Powered Market Prediction System

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import pickle
import json
from pathlib import Path

# Neural Network and ML imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("⚠️ Neural network libraries not available. Install tensorflow and sklearn for full functionality.")

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions"""
    PRICE_DIRECTION = "price_direction"      # Up/Down prediction
    PRICE_TARGET = "price_target"            # Specific price target
    VOLATILITY = "volatility"                # Volatility prediction
    TREND_STRENGTH = "trend_strength"        # Trend strength (0-1)
    REVERSAL_PROBABILITY = "reversal_prob"   # Reversal probability
    SUPPORT_RESISTANCE = "support_resistance" # S/R levels
    BREAKOUT_PROBABILITY = "breakout_prob"   # Breakout probability
    MOMENTUM_SHIFT = "momentum_shift"        # Momentum change
    LIQUIDITY_FLOW = "liquidity_flow"        # Liquidity direction
    INSTITUTIONAL_BIAS = "institutional_bias" # Institutional sentiment

class ModelType(Enum):
    """Types of neural network models"""
    LSTM_BASIC = "lstm_basic"
    LSTM_ATTENTION = "lstm_attention"
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    HYBRID = "hybrid"

class TimeFrame(Enum):
    """Prediction timeframes"""
    SCALP = "1m"      # 1-5 minutes
    SHORT = "5m"      # 5-30 minutes
    MEDIUM = "1h"     # 1-4 hours
    LONG = "4h"       # 4-24 hours
    SWING = "1d"      # 1-7 days

@dataclass
class PredictionResult:
    """Neural network prediction result"""
    timestamp: datetime
    symbol: str
    prediction_type: PredictionType
    timeframe: TimeFrame
    predicted_value: float
    confidence: float  # 0-1
    probability_distribution: Dict[str, float]
    feature_importance: Dict[str, float]
    model_used: ModelType
    accuracy_score: float
    risk_assessment: str
    recommended_action: str
    stop_loss_suggestion: float
    take_profit_suggestion: float
    position_size_multiplier: float

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_predictions: int
    correct_predictions: int

class FeatureEngineer:
    """🔧 Advanced feature engineering for neural networks"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def create_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df = data.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Market microstructure
        df = self._add_microstructure_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Pattern recognition features
        df = self._add_pattern_features(df)
        
        # Cross-timeframe features
        df = self._add_multi_timeframe_features(df)
        
        # Sentiment features (if available)
        df = self._add_sentiment_features(df, symbol)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['open'] - df['close']) / df['close']
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' not in df.columns:
            df['volume'] = 1  # Default volume
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume-Price Trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).fillna(0).cumsum()
        
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['month'] = pd.to_datetime(df['timestamp']).dt.month
            
            # Session indicators
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread (simulated)
        df['spread'] = (df['high'] - df['low']) / df['close'] * 10000  # In pips
        
        # Price impact
        df['price_impact'] = abs(df['close'] - df['open']) / df['close']
        
        # Tick direction
        df['tick_direction'] = np.sign(df['close'].diff())
        
        # Consecutive moves
        df['consecutive_up'] = (df['tick_direction'] == 1).astype(int).groupby(
            (df['tick_direction'] != df['tick_direction'].shift()).cumsum()).cumsum()
        df['consecutive_down'] = (df['tick_direction'] == -1).astype(int).groupby(
            (df['tick_direction'] != df['tick_direction'].shift()).cumsum()).cumsum()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Average True Range
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_ratio'] = df['tr'] / df['atr']
        
        # Volatility measures
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].rolling(period).std()
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df['close']
        
        # GARCH-like volatility
        returns = df['close'].pct_change().dropna()
        df['garch_vol'] = returns.rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # Williams %R
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        # Commodity Channel Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        # Candlestick patterns (simplified)
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        
        # Doji
        df['doji'] = (body / (df['high'] - df['low']) < 0.1).astype(int)
        
        # Hammer
        df['hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.5)).astype(int)
        
        # Shooting star
        df['shooting_star'] = ((upper_shadow > body * 2) & (lower_shadow < body * 0.5)).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) & 
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) & 
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)
        
        return df
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe features"""
        # Higher timeframe trends (simulated by resampling)
        try:
            # Daily trend
            daily_close = df['close'].resample('D').last().ffill()
            daily_trend = (daily_close > daily_close.shift(1)).astype(int)
            df['daily_trend'] = daily_trend.reindex(df.index, method='ffill')
            
            # 4H trend
            h4_close = df['close'].resample('4H').last().ffill()
            h4_trend = (h4_close > h4_close.shift(1)).astype(int)
            df['h4_trend'] = h4_trend.reindex(df.index, method='ffill')
            
        except:
            # Fallback if resampling fails
            df['daily_trend'] = 0
            df['h4_trend'] = 0
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment features (placeholder)"""
        # These would come from news sentiment, social media, etc.
        df['news_sentiment'] = 0.5  # Neutral
        df['social_sentiment'] = 0.5  # Neutral
        df['institutional_sentiment'] = 0.5  # Neutral
        
        return df
    
    def prepare_sequences(self, data: pd.DataFrame, sequence_length: int = 60, 
                         target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        # Remove non-numeric columns and handle NaN
        numeric_data = data.select_dtypes(include=[np.number]).fillna(method='ffill').fillna(0)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Store scaler for later use
        self.scalers['main'] = scaler
        self.feature_names = numeric_data.columns.tolist()
        
        # Create sequences
        X, y = [], []
        target_idx = numeric_data.columns.get_loc(target_column)
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)

class NeuralPredictionEngine:
    """🧠 Advanced Neural Network Prediction Engine
    
    Features:
    - Multiple neural network architectures
    - Ensemble modeling
    - Real-time prediction
    - Model performance tracking
    - Feature importance analysis
    - Automated model selection
    - Risk-adjusted predictions
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.model_performance = {}
        self.feature_engineer = FeatureEngineer()
        
        # Prediction history
        self.prediction_history = deque(maxlen=1000)
        self.model_accuracy = defaultdict(list)
        
        # Configuration
        self.sequence_length = 60
        self.prediction_horizon = 1
        self.ensemble_weights = {}
        
        if not NEURAL_AVAILABLE:
            logger.warning("Neural network libraries not available. Limited functionality.")
        
        logger.info("🧠 Neural Prediction Engine initialized")
    
    def create_lstm_model(self, input_shape: Tuple, model_type: ModelType = ModelType.LSTM_BASIC) -> Any:
        """Create LSTM model architecture"""
        if not NEURAL_AVAILABLE:
            return None
        
        if model_type == ModelType.LSTM_BASIC:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
        
        elif model_type == ModelType.LSTM_ATTENTION:
            # LSTM with attention mechanism
            inputs = tf.keras.Input(shape=input_shape)
            lstm_out = LSTM(50, return_sequences=True)(inputs)
            attention = MultiHeadAttention(num_heads=4, key_dim=50)(lstm_out, lstm_out)
            lstm_out2 = LSTM(50)(attention)
            dropout = Dropout(0.2)(lstm_out2)
            dense = Dense(25, activation='relu')(dropout)
            outputs = Dense(1)(dense)
            model = Model(inputs=inputs, outputs=outputs)
        
        elif model_type == ModelType.CNN_LSTM:
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
        
        else:
            # Default to basic LSTM
            model = self.create_lstm_model(input_shape, ModelType.LSTM_BASIC)
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, symbol: str, data: pd.DataFrame, model_type: ModelType = ModelType.LSTM_BASIC,
                   prediction_type: PredictionType = PredictionType.PRICE_DIRECTION) -> bool:
        """Train neural network model"""
        try:
            if not NEURAL_AVAILABLE:
                logger.warning("Neural network libraries not available")
                return False
            
            # Feature engineering
            features_df = self.feature_engineer.create_features(data, symbol)
            
            # Prepare target variable based on prediction type
            if prediction_type == PredictionType.PRICE_DIRECTION:
                features_df['target'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
            elif prediction_type == PredictionType.PRICE_TARGET:
                features_df['target'] = features_df['close'].shift(-1)
            elif prediction_type == PredictionType.VOLATILITY:
                features_df['target'] = features_df['close'].rolling(20).std().shift(-1)
            else:
                features_df['target'] = features_df['close'].shift(-1)
            
            # Remove rows with NaN targets
            features_df = features_df.dropna()
            
            if len(features_df) < 100:
                logger.warning(f"Insufficient data for training: {len(features_df)} rows")
                return False
            
            # Prepare sequences
            X, y = self.feature_engineer.prepare_sequences(
                features_df.drop('target', axis=1), 
                self.sequence_length, 
                'close'
            )
            
            if len(X) == 0:
                logger.warning("No sequences created")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Create and train model
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOST]:
                # Traditional ML models
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                
                if model_type == ModelType.RANDOM_FOREST:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train_flat, y_train)
                predictions = model.predict(X_test_flat)
                
            else:
                # Neural network models
                model = self.create_lstm_model(input_shape, model_type)
                
                # Callbacks
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                predictions = model.predict(X_test)
            
            # Calculate performance metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            # Store model and performance
            model_key = f"{symbol}_{model_type.value}_{prediction_type.value}"
            self.models[model_key] = model
            
            performance = ModelPerformance(
                model_name=model_key,
                accuracy=1 - (mse / np.var(y_test)) if np.var(y_test) > 0 else 0,
                precision=0,  # Would need classification metrics
                recall=0,
                f1_score=0,
                mse=mse,
                mae=mae,
                sharpe_ratio=0,  # Would need returns data
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                total_predictions=len(y_test),
                correct_predictions=0
            )
            
            self.model_performance[model_key] = performance
            
            # Save model
            model_path = self.models_dir / f"{model_key}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model {model_key} trained successfully. MSE: {mse:.6f}, MAE: {mae:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, symbol: str, data: pd.DataFrame, 
               prediction_type: PredictionType = PredictionType.PRICE_DIRECTION,
               timeframe: TimeFrame = TimeFrame.SHORT) -> Optional[PredictionResult]:
        """🎯 Make prediction using trained models"""
        try:
            # Find best model for this prediction
            model_key = self._find_best_model(symbol, prediction_type)
            
            if not model_key or model_key not in self.models:
                logger.warning(f"No trained model found for {symbol} {prediction_type.value}")
                return None
            
            model = self.models[model_key]
            
            # Feature engineering
            features_df = self.feature_engineer.create_features(data, symbol)
            
            # Prepare input sequence
            if len(features_df) < self.sequence_length:
                logger.warning(f"Insufficient data for prediction: {len(features_df)} rows")
                return None
            
            # Get last sequence
            numeric_data = features_df.select_dtypes(include=[np.number]).fillna(method='ffill').fillna(0)
            
            if 'main' in self.feature_engineer.scalers:
                scaler = self.feature_engineer.scalers['main']
                scaled_data = scaler.transform(numeric_data)
            else:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(numeric_data)
            
            # Create input sequence
            X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            if hasattr(model, 'predict'):
                if len(X.shape) == 3:  # Neural network
                    prediction = model.predict(X, verbose=0)[0][0]
                else:  # Traditional ML
                    X_flat = X.reshape(1, -1)
                    prediction = model.predict(X_flat)[0]
            else:
                logger.error(f"Model {model_key} does not have predict method")
                return None
            
            # Calculate confidence based on model performance
            performance = self.model_performance.get(model_key)
            confidence = 1 - performance.mse if performance else 0.5
            confidence = max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95
            
            # Generate probability distribution
            prob_dist = self._generate_probability_distribution(prediction, prediction_type)
            
            # Feature importance (simplified)
            feature_importance = self._calculate_feature_importance(model, model_key)
            
            # Risk assessment
            risk_assessment = self._assess_prediction_risk(prediction, confidence, data)
            
            # Trading recommendations
            action, stop_loss, take_profit, size_multiplier = self._generate_trading_recommendations(
                prediction, prediction_type, confidence, data
            )
            
            result = PredictionResult(
                timestamp=datetime.now(),
                symbol=symbol,
                prediction_type=prediction_type,
                timeframe=timeframe,
                predicted_value=float(prediction),
                confidence=confidence,
                probability_distribution=prob_dist,
                feature_importance=feature_importance,
                model_used=ModelType(model_key.split('_')[1]),
                accuracy_score=performance.accuracy if performance else 0,
                risk_assessment=risk_assessment,
                recommended_action=action,
                stop_loss_suggestion=stop_loss,
                take_profit_suggestion=take_profit,
                position_size_multiplier=size_multiplier
            )
            
            # Store prediction
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def predict_ensemble(self, symbol: str, data: pd.DataFrame,
                        prediction_type: PredictionType = PredictionType.PRICE_DIRECTION,
                        timeframe: TimeFrame = TimeFrame.SHORT) -> Optional[PredictionResult]:
        """🎯 Make ensemble prediction using multiple models"""
        try:
            # Find all relevant models
            relevant_models = [
                key for key in self.models.keys() 
                if symbol in key and prediction_type.value in key
            ]
            
            if not relevant_models:
                return self.predict(symbol, data, prediction_type, timeframe)
            
            predictions = []
            weights = []
            
            # Get predictions from all models
            for model_key in relevant_models:
                try:
                    # Temporarily set the model for individual prediction
                    temp_result = self.predict(symbol, data, prediction_type, timeframe)
                    if temp_result:
                        predictions.append(temp_result.predicted_value)
                        
                        # Weight by model performance
                        performance = self.model_performance.get(model_key)
                        weight = performance.accuracy if performance else 0.5
                        weights.append(weight)
                except:
                    continue
            
            if not predictions:
                return None
            
            # Calculate weighted ensemble prediction
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_prediction = np.average(predictions, weights=weights)
            ensemble_confidence = np.average([w for w in weights])  # Average confidence
            
            # Create ensemble result
            result = PredictionResult(
                timestamp=datetime.now(),
                symbol=symbol,
                prediction_type=prediction_type,
                timeframe=timeframe,
                predicted_value=float(ensemble_prediction),
                confidence=ensemble_confidence,
                probability_distribution=self._generate_probability_distribution(ensemble_prediction, prediction_type),
                feature_importance={},  # Combined feature importance
                model_used=ModelType.ENSEMBLE,
                accuracy_score=ensemble_confidence,
                risk_assessment=self._assess_prediction_risk(ensemble_prediction, ensemble_confidence, data),
                recommended_action="",
                stop_loss_suggestion=0,
                take_profit_suggestion=0,
                position_size_multiplier=1.0
            )
            
            # Generate recommendations
            action, stop_loss, take_profit, size_multiplier = self._generate_trading_recommendations(
                ensemble_prediction, prediction_type, ensemble_confidence, data
            )
            
            result.recommended_action = action
            result.stop_loss_suggestion = stop_loss
            result.take_profit_suggestion = take_profit
            result.position_size_multiplier = size_multiplier
            
            return result
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return None
    
    def _find_best_model(self, symbol: str, prediction_type: PredictionType) -> Optional[str]:
        """Find best performing model for given symbol and prediction type"""
        relevant_models = [
            key for key in self.model_performance.keys()
            if symbol in key and prediction_type.value in key
        ]
        
        if not relevant_models:
            return None
        
        # Sort by accuracy
        best_model = max(relevant_models, 
                        key=lambda x: self.model_performance[x].accuracy)
        
        return best_model
    
    def _generate_probability_distribution(self, prediction: float, 
                                         prediction_type: PredictionType) -> Dict[str, float]:
        """Generate probability distribution for prediction"""
        if prediction_type == PredictionType.PRICE_DIRECTION:
            if prediction > 0.5:
                return {'up': float(prediction), 'down': float(1 - prediction)}
            else:
                return {'up': float(prediction), 'down': float(1 - prediction)}
        else:
            # For continuous predictions, create confidence intervals
            return {
                'prediction': float(prediction),
                'lower_bound': float(prediction * 0.95),
                'upper_bound': float(prediction * 1.05)
            }
    
    def _calculate_feature_importance(self, model: Any, model_key: str) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_names = self.feature_engineer.feature_names
                
                if len(importances) == len(feature_names):
                    return dict(zip(feature_names, importances.tolist()))
            
            # For neural networks, return placeholder
            return {'price_features': 0.3, 'technical_indicators': 0.4, 'volume_features': 0.3}
            
        except:
            return {}
    
    def _assess_prediction_risk(self, prediction: float, confidence: float, 
                              data: pd.DataFrame) -> str:
        """Assess risk level of prediction"""
        if confidence > 0.8:
            return "low"
        elif confidence > 0.6:
            return "moderate"
        else:
            return "high"
    
    def _generate_trading_recommendations(self, prediction: float, prediction_type: PredictionType,
                                        confidence: float, data: pd.DataFrame) -> Tuple[str, float, float, float]:
        """Generate trading recommendations"""
        current_price = data['close'].iloc[-1] if len(data) > 0 else 0
        atr = data['close'].rolling(14).std().iloc[-1] if len(data) >= 14 else 0.001
        
        if prediction_type == PredictionType.PRICE_DIRECTION:
            if prediction > 0.6:  # Strong bullish
                action = "BUY"
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
                size_multiplier = confidence
            elif prediction < 0.4:  # Strong bearish
                action = "SELL"
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
                size_multiplier = confidence
            else:
                action = "HOLD"
                stop_loss = current_price
                take_profit = current_price
                size_multiplier = 0.5
        else:
            action = "ANALYZE"
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2)
            size_multiplier = confidence * 0.8
        
        return action, stop_loss, take_profit, size_multiplier
    
    def update_model_performance(self, symbol: str, prediction_result: PredictionResult, 
                               actual_outcome: float):
        """Update model performance based on actual outcomes"""
        model_key = f"{symbol}_{prediction_result.model_used.value}_{prediction_result.prediction_type.value}"
        
        if model_key in self.model_performance:
            performance = self.model_performance[model_key]
            
            # Update accuracy tracking
            error = abs(prediction_result.predicted_value - actual_outcome)
            self.model_accuracy[model_key].append(error)
            
            # Recalculate performance metrics
            recent_errors = list(self.model_accuracy[model_key])[-100:]  # Last 100 predictions
            performance.mae = np.mean(recent_errors)
            performance.mse = np.mean([e**2 for e in recent_errors])
            performance.total_predictions += 1
            
            # Update accuracy
            if prediction_result.prediction_type == PredictionType.PRICE_DIRECTION:
                correct = (prediction_result.predicted_value > 0.5) == (actual_outcome > 0.5)
                if correct:
                    performance.correct_predictions += 1
                performance.accuracy = performance.correct_predictions / performance.total_predictions
    
    def get_model_summary(self) -> Dict:
        """Get summary of all models and their performance"""
        summary = {
            'total_models': len(self.models),
            'model_performance': {},
            'best_models': {},
            'prediction_history_count': len(self.prediction_history)
        }
        
        # Add performance for each model
        for model_key, performance in self.model_performance.items():
            summary['model_performance'][model_key] = {
                'accuracy': performance.accuracy,
                'mse': performance.mse,
                'mae': performance.mae,
                'total_predictions': performance.total_predictions
            }
        
        # Find best models by prediction type
        prediction_types = set()
        for key in self.model_performance.keys():
            parts = key.split('_')
            if len(parts) >= 3:
                prediction_types.add('_'.join(parts[2:]))
        
        for pred_type in prediction_types:
            relevant_models = [
                (key, perf) for key, perf in self.model_performance.items()
                if pred_type in key
            ]
            if relevant_models:
                best_model = max(relevant_models, key=lambda x: x[1].accuracy)
                summary['best_models'][pred_type] = {
                    'model': best_model[0],
                    'accuracy': best_model[1].accuracy
                }
        
        return summary
    
    def save_models(self):
        """Save all models to disk"""
        try:
            # Save model performance
            performance_path = self.models_dir / "model_performance.json"
            performance_data = {}
            
            for key, perf in self.model_performance.items():
                performance_data[key] = {
                    'model_name': perf.model_name,
                    'accuracy': perf.accuracy,
                    'mse': perf.mse,
                    'mae': perf.mae,
                    'total_predictions': perf.total_predictions,
                    'correct_predictions': perf.correct_predictions
                }
            
            with open(performance_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            logger.info(f"Models and performance saved to {self.models_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load models from disk"""
        try:
            # Load model performance
            performance_path = self.models_dir / "model_performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    performance_data = json.load(f)
                
                for key, data in performance_data.items():
                    self.model_performance[key] = ModelPerformance(
                        model_name=data['model_name'],
                        accuracy=data['accuracy'],
                        precision=0,
                        recall=0,
                        f1_score=0,
                        mse=data['mse'],
                        mae=data['mae'],
                        sharpe_ratio=0,
                        max_drawdown=0,
                        win_rate=0,
                        profit_factor=0,
                        total_predictions=data['total_predictions'],
                        correct_predictions=data['correct_predictions']
                    )
            
            # Load individual models
            for model_file in self.models_dir.glob("*.pkl"):
                model_key = model_file.stem
                try:
                    with open(model_file, 'rb') as f:
                        self.models[model_key] = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Could not load model {model_key}: {e}")
            
            logger.info(f"Loaded {len(self.models)} models from {self.models_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict_market_movement(self, symbol: str, market_data: Dict) -> Dict:
        """🎯 Predict market movement for given symbol and market data
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            market_data: Dictionary containing market data with keys:
                - timestamp: Current timestamp
                - open, high, low, close: Price data
                - volume: Trading volume (optional)
                - Additional technical indicators
        
        Returns:
            Dictionary containing prediction results:
            - direction: 'BUY', 'SELL', or 'HOLD'
            - confidence: Confidence level (0-1)
            - predicted_price: Predicted price target
            - probability: Probability distribution
            - risk_level: Risk assessment
            - stop_loss: Suggested stop loss
            - take_profit: Suggested take profit
        """
        try:
            # Convert market_data to DataFrame format expected by the prediction engine
            if isinstance(market_data, dict):
                # Create a simple DataFrame from the market data
                df_data = {
                    'timestamp': [market_data.get('timestamp', datetime.now())],
                    'open': [market_data.get('open', market_data.get('close', 1.0))],
                    'high': [market_data.get('high', market_data.get('close', 1.0))],
                    'low': [market_data.get('low', market_data.get('close', 1.0))],
                    'close': [market_data.get('close', 1.0)],
                    'volume': [market_data.get('volume', 1000)]
                }
                
                # Add any additional data from market_data
                for key, value in market_data.items():
                    if key not in df_data and isinstance(value, (int, float)):
                        df_data[key] = [value]
                
                data_df = pd.DataFrame(df_data)
            elif hasattr(market_data, 'dtype') and hasattr(market_data, '__len__'):
                # Handle MT5 rates data (numpy structured array)
                try:
                    data_df = pd.DataFrame(market_data)
                    # Convert time column to datetime if it exists
                    if 'time' in data_df.columns:
                        data_df['timestamp'] = pd.to_datetime(data_df['time'], unit='s')
                        data_df.set_index('timestamp', inplace=True)
                    # Ensure required columns exist
                    required_cols = ['open', 'high', 'low', 'close']
                    for col in required_cols:
                        if col not in data_df.columns:
                            data_df[col] = data_df.get('close', 1.0)
                    if 'volume' not in data_df.columns:
                        data_df['volume'] = 1000
                except Exception as e:
                    logger.warning(f"Error converting MT5 data to DataFrame: {e}")
                    # Fallback to simple DataFrame
                    data_df = pd.DataFrame({
                        'timestamp': [datetime.now()],
                        'open': [1.0], 'high': [1.0], 'low': [1.0], 'close': [1.0], 'volume': [1000]
                    })
            else:
                # If market_data is already a DataFrame
                data_df = market_data.copy()
            
            # Try ensemble prediction first, fallback to single model
            prediction_result = self.predict_ensemble(
                symbol=symbol,
                data=data_df,
                prediction_type=PredictionType.PRICE_DIRECTION,
                timeframe=TimeFrame.SHORT
            )
            
            if not prediction_result:
                # Fallback to basic prediction
                prediction_result = self.predict(
                    symbol=symbol,
                    data=data_df,
                    prediction_type=PredictionType.PRICE_DIRECTION,
                    timeframe=TimeFrame.SHORT
                )
            
            if not prediction_result:
                # Return default prediction if no models available
                logger.warning(f"No prediction available for {symbol}, returning default")
                return {
                    'direction': 'HOLD',
                    'confidence': 0.5,
                    'predicted_price': market_data.get('close', 1.0),
                    'probability': {'up': 0.5, 'down': 0.5},
                    'risk_level': 'high',
                    'stop_loss': market_data.get('close', 1.0) * 0.99,
                    'take_profit': market_data.get('close', 1.0) * 1.01,
                    'model_used': 'default',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Convert PredictionResult to expected dictionary format
            result = {
                'direction': prediction_result.recommended_action,
                'confidence': prediction_result.confidence,
                'predicted_price': prediction_result.predicted_value,
                'probability': prediction_result.probability_distribution,
                'risk_level': prediction_result.risk_assessment,
                'stop_loss': prediction_result.stop_loss_suggestion,
                'take_profit': prediction_result.take_profit_suggestion,
                'model_used': prediction_result.model_used.value,
                'accuracy_score': prediction_result.accuracy_score,
                'timestamp': prediction_result.timestamp.isoformat(),
                'symbol': symbol,
                'timeframe': prediction_result.timeframe.value,
                'position_size_multiplier': prediction_result.position_size_multiplier
            }
            
            logger.info(f"Neural prediction for {symbol}: {result['direction']} (confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_market_movement for {symbol}: {e}")
            # Return safe default prediction
            return {
                'direction': 'HOLD',
                'confidence': 0.5,
                'predicted_price': market_data.get('close', 1.0) if isinstance(market_data, dict) else 1.0,
                'probability': {'up': 0.5, 'down': 0.5},
                'risk_level': 'high',
                'stop_loss': (market_data.get('close', 1.0) if isinstance(market_data, dict) else 1.0) * 0.99,
                'take_profit': (market_data.get('close', 1.0) if isinstance(market_data, dict) else 1.0) * 1.01,
                'model_used': 'error_fallback',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

def create_neural_prediction_engine(models_dir: str = "models") -> NeuralPredictionEngine:
    """Factory function to create neural prediction engine"""
    return NeuralPredictionEngine(models_dir)