import pandas as pd
import numpy as np
import talib
import logging
import asyncio
import datetime
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

class LiveMAStrategy:
    def __init__(self, client, symbols, timeframes, config=None):
        self.client = client
        self.symbols = symbols
        self.timeframes = timeframes
        self.config = config or {}
        self.leverage = self.config.get('leverage', 20)
        self.tp_roi = self.config.get('tp_roi', 3.5)
        self.sl_roi = self.config.get('sl_roi', 2.8)
        self.min_score = self.config.get('min_score', 3)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 300)
        self.mode = self.config.get('mode', 'maker')
        self.price_offset_pct = self.config.get('price_offset_pct', 0.1)
        self.entry_timeout_sec = self.config.get('entry_timeout_sec', 5)
        self.maintenance_margin_rate = 0.005
        self.console = Console()
        self.in_position = {symbol: False for symbol in symbols}
        self.position_side = {symbol: None for symbol in symbols}
        self.entry_price = {symbol: 0.0 for symbol in symbols}
        self.sl_price = {symbol: 0.0 for symbol in symbols}
        self.tp_price = {symbol: 0.0 for symbol in symbols}
        self.quantity = {symbol: 0.0 for symbol in symbols}
        self.latest_close = {symbol: 0.0 for symbol in symbols}
        self.data = {symbol: {tf: pd.DataFrame() for tf in timeframes} for symbol in symbols}
        self.last_signal = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.unrealized_pnl = {symbol: 0.0 for symbol in symbols}
        self.realized_pnl = {symbol: 0.0 for symbol in symbols}
        self.funding_fee = {symbol: 0.0 for symbol in symbols}
        self.commission = {symbol: 0.0 for symbol in symbols}
        self.margin_used = {symbol: 0.0 for symbol in symbols}
        self.liquidation_price = {symbol: 0.0 for symbol in symbols}
        self.margin_ratio = {symbol: 0.0 for symbol in symbols}
        self.position_timeframe = {symbol: None for symbol in symbols}
        self.update_positions_task = None
        self.monitor_positions_task = None
        self.performance_window = 30
        self.min_win_rate = 0.45
        self.disable_timeframes = {symbol: set() for symbol in symbols}
        self.last_adaptation = datetime.datetime.utcnow()
        self.debug_signals = True
        self.force_entry = False
        self.last_closed_time = {symbol: None for symbol in symbols}

        # Timeframes for multi-timeframe confirmation
        self.long_timeframes = ["1h", "2h", "4h", "6h", "8h", "12h", "1d"]
        self.short_timeframes = ["5m", "15m", "30m"]

    def check_cooldown(self, symbol):
        if self.last_closed_time.get(symbol) is not None:
            elapsed = (datetime.datetime.utcnow() - self.last_closed_time[symbol]).total_seconds()
            if elapsed < self.cooldown_seconds:
                if self.debug_signals:
                    logger.info(f"Cooldown active for {symbol}: waiting {self.cooldown_seconds - elapsed:.0f} seconds")
                return False
        return True

    def update_close_time(self, symbol):
        self.last_closed_time[symbol] = datetime.datetime.utcnow()

    async def async_init(self):
        for symbol in self.symbols:
            try:
                symbol_clean = symbol.replace('/', '')
                response = await self.client.exchange.fapiPrivatePostLeverage({
                    'symbol': symbol_clean,
                    'leverage': self.leverage
                })
                logger.info(f"Set leverage {self.leverage}x for {symbol}: {response}")
            except Exception as e:
                logger.error(f"Failed to set leverage for {symbol}: {e}")
        await self.fetch_initial_history()
        self.update_positions_task = asyncio.create_task(self.update_positions_loop())
        self.monitor_positions_task = asyncio.create_task(self.monitor_positions_loop())

    async def fetch_initial_history(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                limit = {
                    '5m': 1200,
                    '15m': 800,
                    '30m': 480,
                    '1h': 720,
                    '2h': 360,
                    '4h': 180,
                    '6h': 120,
                    '8h': 100,
                    '12h': 80,
                    '1d': 60
                }.get(tf, 100)
                try:
                    ohlcv = await self.client.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open','high','low','close','volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    self.data[symbol][tf] = df
                    logger.info(f"Fetched {len(df)} candles for {symbol} {tf}")
                except Exception as e:
                    logger.error(f"Failed to fetch history for {symbol} {tf}: {e}")

    async def update_positions_loop(self):
        while True:
            try:
                await self.fetch_positions()
            except Exception as e:
                logger.error(f"Failed to update positions: {e}")
            await asyncio.sleep(3)

    async def monitor_positions_loop(self):
        while True:
            try:
                for symbol in self.symbols:
                    await self.evaluate_strategy(symbol)
                    logger.debug(f"Periodic evaluation triggered for {symbol}")
            except Exception as e:
                logger.error(f"Failed to monitor positions: {e}")
            await asyncio.sleep(5)

    async def fetch_positions(self):
        try:
            positions = await self.client.exchange.fapiPrivateV2GetPositionRisk()
            for pos in positions:
                symbol = pos.get('symbol')
                if symbol not in self.symbols:
                    continue
                qty = float(pos.get('positionAmt', 0) or 0)
                entry = float(pos.get('entryPrice', 0) or 0)
                mark = float(pos.get('markPrice', entry) or entry)
                un_pnl = float(pos.get('unRealizedProfit', 0) or 0)
                leverage = float(pos.get('leverage', self.leverage) or self.leverage)
                notional = abs(qty * mark)
                margin = notional / leverage if leverage else 0.0
                liq_price = float(pos.get('liquidationPrice', 0) or 0)
                margin_ratio = abs(un_pnl) / margin * 100 if margin else 0.0
                self.in_position[symbol] = abs(qty) > 0
                self.position_side[symbol] = "long" if qty > 0 else ("short" if qty < 0 else None)
                if self.in_position[symbol] is False and self.position_side[symbol] is not None:
                    logger.warning(f"Detected manual close of position for {symbol}. Resetting state.")
                    self.position_side[symbol] = None
                    self.entry_price[symbol] = 0.0
                    self.sl_price[symbol] = 0.0
                    self.tp_price[symbol] = 0.0
                    self.quantity[symbol] = 0.0
                    self.position_timeframe[symbol] = None

                self.entry_price[symbol] = entry if abs(qty) > 0 else 0.0
                self.quantity[symbol] = abs(qty)
                self.latest_close[symbol] = mark
                self.unrealized_pnl[symbol] = un_pnl
                self.margin_used[symbol] = margin
                self.liquidation_price[symbol] = liq_price
                self.margin_ratio[symbol] = margin_ratio
                funding = await self.fetch_funding_fee(symbol)
                self.funding_fee[symbol] = funding
                commission = await self.fetch_commission(symbol)
                self.commission[symbol] = commission
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")

    async def fetch_funding_fee(self, symbol):
        try:
            rows = await self.client.exchange.fapiPrivateGetIncome(
                {"symbol": symbol, "incomeType": "FUNDING_FEE", "limit": 10}
            )
            funding = sum(float(x['income']) for x in rows)
            return funding
        except Exception as e:
            logger.warning(f"Failed to fetch funding for {symbol}: {e}")
            return 0.0

    async def fetch_commission(self, symbol):
        try:
            trades = await self.client.exchange.fapiPrivateGetUserTrades(
                {"symbol": symbol, "limit": 10}
            )
            commission = sum(float(t['commission']) for t in trades)
            return commission
        except Exception as e:
            logger.warning(f"Failed to fetch commission for {symbol}: {e}")
            return 0.0

    async def process_timeframe_data(self, symbol, timeframe, df):
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if timeframe not in self.timeframes:
            logger.warning(f"Received invalid timeframe {timeframe} for {symbol}, expected one of {self.timeframes}")
            return
        if df is None or df.empty:
            logger.warning(f"Empty OHLCV data for {symbol} {timeframe}")
            return
        
        logger.debug(f"Received OHLCV data for {symbol} {timeframe}: columns={df.columns.tolist()}, data={df.to_dict(orient='records')}")
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols} in OHLCV data for {symbol} {timeframe}")
            return
        
        try:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            self.data[symbol][timeframe] = pd.concat([self.data[symbol][timeframe], df]).drop_duplicates(subset=['timestamp']).tail(100)
            self.latest_close[symbol] = df['close'].iloc[-1]
            logger.info(f"Processed WebSocket data for {symbol} {timeframe}: timestamp={df['timestamp'].iloc[-1]}, close={self.latest_close[symbol]}")
            await self.evaluate_strategy(symbol)
            logger.info(f"Evaluated strategy for {symbol} {timeframe}: signals updated")
        except Exception as e:
            logger.warning(f"Invalid OHLCV data format for {symbol} {timeframe}: {e}")

    async def process_tick(self, symbol, price):
        self.latest_close[symbol] = price

    def get_signal_for_timeframe(self, symbol, timeframe):
        df = self.data[symbol][timeframe]
        if len(df) < 60:
            logger.debug(f"Not enough data for {symbol} {timeframe} to check signal: {len(df)} candles")
            return None

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        try:
            stoch_rsi_k, stoch_rsi_d = talib.STOCHRSI(close, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
        except Exception:
            stoch_rsi_k = pd.Series(np.nan, index=close.index)
            stoch_rsi_d = pd.Series(np.nan, index=close.index)
        stoch_cross_long = False
        stoch_cross_short = False
        if len(stoch_rsi_k) > 2 and not np.isnan(stoch_rsi_k.iloc[-2]) and not np.isnan(stoch_rsi_d.iloc[-2]):
            if stoch_rsi_k.iloc[-2] < stoch_rsi_d.iloc[-2] and stoch_rsi_k.iloc[-1] > stoch_rsi_d.iloc[-1]:
                stoch_cross_long = True
            if stoch_rsi_k.iloc[-2] > stoch_rsi_d.iloc[-2] and stoch_rsi_k.iloc[-1] < stoch_rsi_d.iloc[-1]:
                stoch_cross_short = True

        ema9 = talib.EMA(close, timeperiod=9)
        ema21 = talib.EMA(close, timeperiod=21)
        ema13 = talib.EMA(close, timeperiod=13)
        ema55 = talib.EMA(close, timeperiod=55)
        ema9_21_cross_long = ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]
        ema9_21_cross_short = ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]
        ema13_55_cross_long = ema13.iloc[-2] < ema55.iloc[-2] and ema13.iloc[-1] > ema55.iloc[-1]
        ema13_55_cross_short = ema13.iloc[-2] > ema55.iloc[-2] and ema13.iloc[-1] < ema55.iloc[-1]
        ema_cross_confirm_long = ema9_21_cross_long and ema13_55_cross_long
        ema_cross_confirm_short = ema9_21_cross_short and ema13_55_cross_short

        window = 20
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-9)
        high_vol = vol_z.iloc[-1] > 2
        low_vol = vol_z.iloc[-1] < -2

        adx = talib.ADX(high, low, close, timeperiod=14)
        tf_adx_thresholds = {
            "5m": 15,
            "15m": 20,
            "30m": 22,
            "1h": 25,
            "2h": 25,
            "4h": 30,
            "6h": 30,
            "8h": 32,
            "12h": 35,
            "1d": 37
        }
        adx_thresh = tf_adx_thresholds.get(timeframe, 25)
        adx_strong = adx.iloc[-1] > adx_thresh

        rsi = talib.RSI(close, timeperiod=14)
        rsi_pct_high = np.percentile(rsi.dropna()[-30:], 80)
        rsi_pct_low = np.percentile(rsi.dropna()[-30:], 20)
        rsi_long = rsi.iloc[-1] < rsi_pct_low
        rsi_short = rsi.iloc[-1] > rsi_pct_high

        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macdhist_slope_pos = macdhist.iloc[-1] > 0 and (macdhist.iloc[-1] - macdhist.iloc[-2]) > 0
        macdhist_slope_neg = macdhist.iloc[-1] < 0 and (macdhist.iloc[-1] - macdhist.iloc[-2]) < 0

        try:
            tema = talib.TEMA(close, timeperiod=21)
            tema_slope = tema.iloc[-1] - tema.iloc[-5]
        except Exception:
            tema = pd.Series(np.nan, index=close.index)
            tema_slope = 0
        tema_up = tema_slope > 0
        tema_down = tema_slope < 0

        indicator_long = 0
        indicator_short = 0

        if stoch_cross_long:
            indicator_long += 1
        if stoch_cross_short:
            indicator_short += 1

        if ema_cross_confirm_long:
            indicator_long += 1
        if ema_cross_confirm_short:
            indicator_short += 1

        if high_vol:
            indicator_long += 1
        if low_vol:
            indicator_short += 1

        if adx_strong and tema_up:
            indicator_long += 1
        if adx_strong and tema_down:
            indicator_short += 1

        if rsi_long:
            indicator_long += 1
        if rsi_short:
            indicator_short += 1

        if macdhist_slope_pos:
            indicator_long += 1
        if macdhist_slope_neg:
            indicator_short += 1

        if tema_up:
            indicator_long += 1
        if tema_down:
            indicator_short += 1

        signal = None
        if not self.check_cooldown(symbol):
            logger.debug(f"Cooldown active for {symbol} {timeframe}: skipping entry signals")
            return None

        if indicator_long >= self.min_score:
            logger.info(f"Buy conditions met for {symbol} {timeframe} with score {indicator_long}")
            signal = "long"
        elif indicator_short >= self.min_score:
            logger.info(f"Sell conditions met for {symbol} {timeframe} with score {indicator_short}")
            signal = "short"

        if self.force_entry:
            signal = "long"

        logger.debug(
            f"Signal for {symbol} {timeframe}: {signal} "
            f"(StochRSI(L:{stoch_cross_long},S:{stoch_cross_short}), "
            f"EMA Cross(L:{ema_cross_confirm_long},S:{ema_cross_confirm_short}), "
            f"VolZ(L:{high_vol},S:{low_vol}), "
            f"ADX(L:{adx_strong and tema_up},S:{adx_strong and tema_down}), "
            f"RSI(L:{rsi_long},S:{rsi_short}), "
            f"MACD(L:{macdhist_slope_pos},S:{macdhist_slope_neg}), "
            f"TEMA(L:{tema_up},S:{tema_down}), "
            f"Long count: {indicator_long}, Short count: {indicator_short})"
        )
        return signal

    def check_multi_timeframe_signal(self, symbol):
        long_timeframes = self.long_timeframes
        short_timeframes = self.short_timeframes

        main_signal = None
        for tf in long_timeframes:
            sig = self.get_signal_for_timeframe(symbol, tf)
            logger.debug(f"Main signal check for {symbol} {tf}: {sig}")
            if sig is not None:
                main_signal = sig
                break
        if not main_signal:
            logger.debug(f"No main signal for {symbol} from long timeframes")
            return None

        confirm = False
        for tf in short_timeframes:
            sig = self.get_signal_for_timeframe(symbol, tf)
            logger.debug(f"Confirmation signal check for {symbol} {tf}: {sig}")
            if sig == main_signal:
                confirm = True
                break

        if confirm:
            logger.debug(f"Confirmed {main_signal} signal for {symbol}")
            return main_signal
        logger.debug(f"No confirmation for {main_signal} signal for {symbol}")
        return None

    async def evaluate_strategy(self, symbol):
        logger.debug(f"Evaluating strategy for {symbol}")
        self.automate_learning(symbol)

        signal = self.check_multi_timeframe_signal(symbol)
        if signal and not self.in_position[symbol]:
            qty = await self.calc_order_qty(symbol, self.latest_close[symbol])
            logger.info(f"Placing {signal.upper()} order for {symbol} at price {self.latest_close[symbol]}")
            await self.open_position(symbol, signal, self.latest_close[symbol], qty)
            for tf in self.long_timeframes:
                if self.get_signal_for_timeframe(symbol, tf) == signal:
                    self.position_timeframe[symbol] = tf
                    break
            logger.info(f"{datetime.datetime.now()} Entry on {symbol} (multi-TF): {signal.upper()}")

        if self.in_position[symbol]:
            tf = self.position_timeframe[symbol]
            signal = self.get_signal_for_timeframe(symbol, tf) if tf else None
            side = self.position_side[symbol]
            price = self.latest_close[symbol]

            logger.info(
                f"Evaluating position for {symbol} ({side}, timeframe={tf}): "
                f"current_price={price}, entry={self.entry_price[symbol]}, "
                f"sl={self.sl_price[symbol]}, tp={self.tp_price[symbol]}, signal={signal}"
            )

            if side == "long" and price >= self.entry_price[symbol] * 1.015:
                new_sl = max(self.sl_price[symbol], self.entry_price[symbol])
                if new_sl != self.sl_price[symbol]:
                    logger.info(f"[{symbol}] Break-even activated (LONG). SL adjusted to {new_sl}")
                    self.sl_price[symbol] = new_sl
            elif side == "short" and price <= self.entry_price[symbol] * 0.985:
                new_sl = min(self.sl_price[symbol], self.entry_price[symbol])
                if new_sl != self.sl_price[symbol]:
                    logger.info(f"[{symbol}] Break-even activated (SHORT). SL adjusted to {new_sl}")
                    self.sl_price[symbol] = new_sl

            if side == "long" and price >= self.entry_price[symbol] * 1.03:
                new_sl = max(self.sl_price[symbol], price * 0.995)
                if new_sl != self.sl_price[symbol]:
                    logger.info(f"[{symbol}] Trailing stop activated (LONG). SL adjusted to {new_sl}")
                    self.sl_price[symbol] = new_sl
            elif side == "short" and price <= self.entry_price[symbol] * 0.97:
                new_sl = min(self.sl_price[symbol], price * 1.005)
                if new_sl != self.sl_price[symbol]:
                    logger.info(f"[{symbol}] Trailing stop activated (SHORT). SL adjusted to {new_sl}")
                    self.sl_price[symbol] = new_sl

            should_close = False
            reason = ""
            if side == "long":
                if price <= self.sl_price[symbol]:
                    should_close = True
                    reason = "hit stop loss"
                elif price >= self.tp_price[symbol]:
                    should_close = True
                    reason = "hit take profit"
                elif signal == "short":
                    should_close = True
                    reason = "opposite signal"
            elif side == "short":
                if price >= self.sl_price[symbol]:
                    should_close = True
                    reason = "hit stop loss"
                elif price <= self.tp_price[symbol]:
                    should_close = True
                    reason = "hit take profit"
                elif signal == "long":
                    should_close = True
                    reason = "opposite signal"

            if should_close:
                logger.info(f"Closing position for {symbol}: {reason}")
                await self.close_position(symbol, price)
            else:
                logger.info(f"Position for {symbol} remains open: price={price}, sl={self.sl_price[symbol]}, tp={self.tp_price[symbol]}, signal={signal}")

            margin_ratio = self.margin_ratio.get(symbol, 0.0)
            if margin_ratio >= 80:
                logger.warning(f"{symbol} margin ratio {margin_ratio:.2f}%: CLOSE TO LIQUIDATION!")

    async def calc_order_qty(self, symbol, price):
        try:
            balance = await self.client.exchange.fetch_balance()
            usdt_bal = balance['total']['USDT']
            adaptive_risk = self.get_adaptive_risk(symbol)
            risk_amt = usdt_bal * adaptive_risk
            qty = (risk_amt * self.leverage) / price
            qty = round(qty, 3)
            logger.info(f"Calculated qty for {symbol}: {qty} (risk_amt={risk_amt}, price={price}, leverage={self.leverage})")
            return qty
        except Exception as e:
            logger.error(f"Failed to calc qty: {e}")
            return 0.0

    def get_adaptive_risk(self, symbol):
        win_rate = self.get_win_rate(symbol)
        min_risk = 0.005
        max_risk = 0.02
        base_risk = 0.01
        if win_rate is None:
            return base_risk
        if win_rate < 0.5:
            return max(min_risk, base_risk * 0.5)
        elif win_rate > 0.65:
            return min(max_risk, base_risk * 1.5)
        return base_risk

    async def open_position(self, symbol, side, price, qty):
        if qty <= 0:
            logger.warning(f"Qty zero, cannot open {side} for {symbol}")
            return

        max_attempts = 2  # Retry up to 2 times
        attempt = 1

        while attempt <= max_attempts:
            try:
                order_side = 'buy' if side == "long" else 'sell'

                # Cancel any existing open orders for the symbol
                try:
                    open_orders = await self.client.exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        await self.client.exchange.cancel_order(order['id'], symbol)
                        logger.info(f"Canceled existing order for {symbol}: order_id={order['id']}")
                except Exception as e:
                    logger.warning(f"Failed to cancel open orders for {symbol}: {e}")

                # Calculate limit price with adaptive offset
                if self.mode == "maker":
                    df = self.data[symbol].get('5m', pd.DataFrame())
                    if len(df) >= 20:
                        volatility = df['close'].tail(20).pct_change().std() * 100
                        offset_pct = min(max(volatility * 0.5, 0.03), 0.2)
                    else:
                        offset_pct = self.price_offset_pct

                    # Reduce offset for retries to increase fill chance
                    if attempt > 1:
                        offset_pct *= 0.5  # Halve the offset for retry
                        logger.info(f"Retry attempt {attempt} for {symbol}: reduced offset_pct to {offset_pct}")

                    limit_price = price * (1 - offset_pct / 100) if side == "long" else price * (1 + offset_pct / 100)
                    logger.info(f"Placing MAKER {side.upper()} LIMIT order for {symbol} (attempt {attempt}): qty={qty}, limit_price={limit_price:.4f}, offset_pct={offset_pct}")

                    order = await self.client.exchange.create_limit_order(symbol, order_side, qty, limit_price)

                    wait_time = self.entry_timeout_sec
                    interval = 0.2
                    checks = int(wait_time / interval)

                    for _ in range(checks):
                        await asyncio.sleep(interval)
                        order_status = await self.client.exchange.fetch_order(order['id'], symbol)
                        if order_status['status'] == 'closed':
                            logger.info(f"Order filled for {symbol}: order_id={order['id']}")
                            break
                    else:
                        await self.client.exchange.cancel_order(order['id'], symbol)
                        logger.warning(f"{symbol} maker entry not filled (attempt {attempt}). Order canceled.")
                        attempt += 1
                        continue  # Retry with a new order

                    # Order filled, update position
                    self.in_position[symbol] = True
                    self.position_side[symbol] = side
                    self.entry_price[symbol] = price
                    lev = self.leverage
                    tp_roi = self.tp_roi
                    sl_roi = self.sl_roi

                    if side == "long":
                        self.tp_price[symbol] = price * (1 + tp_roi / 100 / lev)
                        self.sl_price[symbol] = price * (1 - sl_roi / 100 / lev)
                    else:
                        self.tp_price[symbol] = price * (1 - tp_roi / 100 / lev)
                        self.sl_price[symbol] = price * (1 + sl_roi / 100 / lev)

                    self.quantity[symbol] = qty
                    self.log_trade(symbol, self.position_timeframe[symbol] or "unknown", side, price, "OPEN", 0.0)
                    return  # Exit after successful fill

                else:
                    logger.info(f"Placing TAKER {side.upper()} MARKET order for {symbol}: qty={qty}")
                    order = await self.client.exchange.create_market_order(symbol, order_side, qty)

                    self.in_position[symbol] = True
                    self.position_side[symbol] = side
                    self.entry_price[symbol] = price
                    lev = self.leverage
                    tp_roi = self.tp_roi
                    sl_roi = self.sl_roi

                    if side == "long":
                        self.tp_price[symbol] = price * (1 + tp_roi / 100 / lev)
                        self.sl_price[symbol] = price * (1 - sl_roi / 100 / lev)
                    else:
                        self.tp_price[symbol] = price * (1 - tp_roi / 100 / lev)
                        self.sl_price[symbol] = price * (1 + sl_roi / 100 / lev)

                    self.quantity[symbol] = qty
                    self.log_trade(symbol, self.position_timeframe[symbol] or "unknown", side, price, "OPEN", 0.0)
                    return

            except Exception as e:
                logger.error(f"Failed to place order for {symbol} (attempt {attempt}): {e}")
                attempt += 1
                if attempt <= max_attempts:
                    await asyncio.sleep(1)  # Brief pause before retry
                continue

        logger.warning(f"Failed to open {side} position for {symbol} after {max_attempts} attempts")

    async def close_position(self, symbol, price):
        try:
            side = self.position_side[symbol]
            qty = self.quantity[symbol]
            if qty <= 0:
                return

            order_side = 'sell' if side == "long" else 'buy'

            if self.mode == "maker":
                limit_price = price * (1 + self.price_offset_pct / 100) if side == "long" else price * (1 - self.price_offset_pct / 100)
                logger.info(f"Placing FAST LIMIT CLOSE {side.upper()} order for {symbol}: qty={qty}, limit_price={limit_price:.4f}")

                order = await self.client.exchange.create_limit_order(symbol, order_side, qty, limit_price)

                wait_time = self.entry_timeout_sec
                interval = 0.2
                attempts = int(wait_time / interval)

                for _ in range(attempts):
                    await asyncio.sleep(interval)
                    order_status = await self.client.exchange.fetch_order(order['id'], symbol)
                    if order_status['status'] == 'closed':
                        break
                else:
                    await self.client.exchange.cancel_order(order['id'], symbol)
                    logger.warning(f"{symbol} close order not filled in time. Canceled.")
                    return

            else:
                logger.info(f"Placing TAKER CLOSE {side.upper()} MARKET order for {symbol}: qty={qty}")
                order = await self.client.exchange.create_market_order(symbol, order_side, qty)

            logger.info(f"Closed {side} {symbol} {qty} @ {price}")
            self.in_position[symbol] = False
            self.position_side[symbol] = None
            self.entry_price[symbol] = 0.0
            self.sl_price[symbol] = 0.0
            self.tp_price[symbol] = 0.0
            self.quantity[symbol] = 0.0
            self.realized_pnl[symbol] += self.unrealized_pnl.get(symbol, 0.0)
            self.log_trade(symbol, self.position_timeframe[symbol] or "unknown", side, self.entry_price[symbol], price, self.unrealized_pnl.get(symbol, 0.0))
            self.position_timeframe[symbol] = None
            self.update_close_time(symbol)
        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def log_trade(self, symbol, timeframe, side, entry_price, exit_price, pnl):
        result = "win" if isinstance(pnl, (float, int)) and pnl > 0 else "loss"
        try:
            with open("trade_log.csv", "a") as f:
                f.write(f"{datetime.datetime.now()},{symbol},{timeframe},{side},{entry_price},{exit_price},{pnl},{result}\n")
        except Exception as e:
            logger.warning(f"Failed to log trade: {e}")

    def get_recent_trades(self, symbol):
        try:
            df = pd.read_csv("trade_log.csv", header=None, names=[
                "datetime","symbol","timeframe","side","entry","exit","pnl","result"
            ])
            recent = df[df["symbol"] == symbol].tail(self.performance_window)
            return recent
        except Exception:
            return pd.DataFrame()

    def get_win_rate(self, symbol):
        trades = self.get_recent_trades(symbol)
        if trades.empty:
            return None
        return (trades['result'] == "win").mean()

    def automate_learning(self, symbol):
        now = datetime.datetime.utcnow()
        if (now - self.last_adaptation).total_seconds() < 600:
            return
        trades = self.get_recent_trades(symbol)
        if trades.empty:
            return
        grouped = trades.groupby("timeframe")["result"].value_counts().unstack(fill_value=0)
        for tf in self.timeframes:
            wins = grouped.loc[tf]["win"] if tf in grouped.index and "win" in grouped.columns else 0
            total = grouped.loc[tf].sum() if tf in grouped.index else 0
            win_rate = wins / total if total > 0 else 0
            if total >= 8 and win_rate < self.min_win_rate:
                self.disable_timeframes[symbol].add(tf)
                logger.info(f"Disabled {tf} for {symbol} (win rate {win_rate:.2f})")
            elif tf in self.disable_timeframes[symbol] and win_rate >= self.min_win_rate:
                self.disable_timeframes[symbol].remove(tf)
                logger.info(f"Re-enabled {tf} for {symbol} (win rate improved to {win_rate:.2f})")
        self.last_adaptation = now

    def display_status(self):
        table = Table(title=f"Strategy Status ({self.leverage}x Leverage)", show_lines=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Side", style="magenta")
        table.add_column("Entry Price", style="yellow")
        table.add_column("Last Price", style="white")
        table.add_column("Stop Loss", style="red")
        table.add_column("Take Profit", style="blue")
        table.add_column("Qty", style="cyan")
        table.add_column("Unreal. PnL", style="green")
        table.add_column("PnL %", style="green")
        table.add_column("Realiz. PnL", style="green")
        table.add_column("Liq. Price", style="red")
        table.add_column("Margin", style="yellow")
        table.add_column("Mgn %", style="yellow")
        table.add_column("Funding", style="yellow")
        table.add_column("Comm.", style="red")
        table.add_column("Tfs OFF", style="bold red")
        table.add_column("TP PnL", style="green")
        table.add_column("SL PnL", style="red")
        for symbol in self.symbols:
            pos = "IN POSITION" if self.in_position.get(symbol) else "NO POSITION"
            side = self.position_side.get(symbol) or "-"
            entry = self.entry_price.get(symbol, 0.0)
            price = self.latest_close.get(symbol, 0.0)
            sl = self.sl_price.get(symbol, 0.0)
            tp = self.tp_price[symbol]
            qty = self.quantity.get(symbol, 0.0)
            unpnl = self.unrealized_pnl.get(symbol, 0.0)
            margin = self.margin_used.get(symbol, 0.0)
            pnl_pct = f"{(unpnl/margin*100):.2f}%" if margin else "0.00%"
            rpnl = f"{self.realized_pnl.get(symbol, 0.0):.4f}"
            liq = f"{self.liquidation_price.get(symbol, 0.0):.2f}"
            mgn = f"{margin:.2f}"
            mgn_pct = f"{self.margin_ratio.get(symbol, 0.0):.2f}%"
            fund = f"{self.funding_fee.get(symbol, 0.0):.4f}"
            comm = f"{self.commission.get(symbol, 0.0):.4f}"
            tfs_off = ",".join(self.disable_timeframes[symbol]) if self.disable_timeframes[symbol] else "-"

            if qty > 0:
                if side == "long":
                    pnl_tp = (tp - entry) * qty
                    pnl_sl = (sl - entry) * qty
                elif side == "short":
                    pnl_tp = (entry - tp) * qty
                    pnl_sl = (entry - sl) * qty
                else:
                    pnl_tp = pnl_sl = 0.0
            else:
                pnl_tp = pnl_sl = 0.0

            table.add_row(
                symbol, pos, side, f"{entry}", f"{price}", f"{sl}", f"{tp}", f"{qty}", f"{unpnl:.4f}",
                pnl_pct, rpnl, liq, mgn, mgn_pct, fund, comm, tfs_off,
                f"{pnl_tp:.2f}", f"{pnl_sl:.2f}"
            )
        self.console.print(table)

    def enable_force_entry(self):
        self.force_entry = True
        logger.warning("Force entry is ENABLED: bot will always try to enter a long position for testing.")

    def disable_force_entry(self):
        self.force_entry = False
        logger.warning("Force entry is DISABLED: bot will only enter on real signals.")

    def print_signals(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                signal = self.get_signal_for_timeframe(symbol, tf)
                print(f"{symbol} {tf}: {signal}")
