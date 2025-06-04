import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score


def extract_features(df):
    features = pd.DataFrame()
    features['ema_short'] = talib.EMA(df['close'], timeperiod=9)
    features['ema_long'] = talib.EMA(df['close'], timeperiod=21)
    macd, macdsignal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd'] = macd
    features['macdsignal'] = macdsignal
    features['rsi'] = talib.RSI(df['close'], timeperiod=14)
    features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    features['obv'] = talib.OBV(df['close'], df['volume'])
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    features['volume'] = df['volume']

    # Novas features
    closes = df['close']
    features['pct_green_10'] = (closes.diff() > 0).rolling(10).mean()
    features['pct_green_20'] = (closes.diff() > 0).rolling(20).mean()
    features['rel_volume'] = df['volume'] / df['volume'].rolling(14).mean()

    return features.dropna()


def train_model(log_path='trade_log.csv', model_output='model_xgb.pkl'):
    try:
        trades = pd.read_csv(log_path)
        trades = trades.dropna()
        trades = trades[trades['type'] == 'ENTRY']

        X_list = []
        y_list = []

        for _, row in trades.iterrows():
            try:
                df = pd.DataFrame({
                    'open': [row['open']],
                    'high': [row['high']],
                    'low': [row['low']],
                    'close': [row['close']],
                    'volume': [row['volume']]
                })
                df = pd.concat([df] * 150, ignore_index=True)  # Simular série temporal
                feats = extract_features(df)
                if feats.empty:
                    continue
                X_list.append(feats.iloc[-1])
                y_list.append(1 if row['result'].lower() == 'win' else 0)
            except Exception as e:
                print(f"Erro ao processar linha: {e}")

        if not X_list:
            print("Nenhum dado válido para treinar.")
            return

        X = pd.DataFrame(X_list)
        y = np.array(y_list)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        print("--- Validação Cruzada ---")
        print(f"ROC AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")
        print(f"Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")

        model.fit(X, y)
        joblib.dump(model, model_output)
        print(f"Modelo treinado e salvo em: {model_output}")

    except Exception as e:
        print(f"Erro ao treinar modelo: {e}")


if __name__ == '__main__':
    train_model()
