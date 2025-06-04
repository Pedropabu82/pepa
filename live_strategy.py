import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import asyncio
import logging
import talib
from datetime import datetime, timedelta
import traceback
from auto_retrain import train_from_log

logger = logging.getLogger(__name__)

class LiveMAStrategy:
    def __init__(self, client, config):
        self.client = client
        self.config = config
        try:
            self.symbols = list(config['indicators'].keys())
            if not self.symbols:
                raise ValueError("No symbols specified in config['indicators']")
        except KeyError:
            logger.error("Missing 'indicators' key in config.json")
            raise ValueError("Config must include 'indicators' with at least one symbol")
        self.leverage = {symbol: config.get('leverage', 10) for symbol in self.symbols}
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.data = {symbol: {} for symbol in self.symbols}
        self.position_side = {symbol: None for symbol in self.symbols}
        self.entry_price = {symbol: None for symbol in self.symbols}
        self.quantity = {symbol: None for symbol in self.symbols}
        self.unrealized_pnl = {symbol: 0 for symbol in self.symbols}
        self.price_precision = {symbol: 2 for symbol in self.symbols}
        self.quantity_precision = {symbol: 4 for symbol in self.symbols}
        self.cooldown = {symbol: None for symbol in self.symbols}
        self.max_trades = config.get('max_trades_per_day', 5)
        self.daily_trades = {symbol: [] for symbol in self.symbols}
        self.min_qty = {'BTCUSDT': 0.001, 'ETHUSDT': 0.01, 'SOLUSDT': 0.1}
        logger.info(f"Initialized with symbols: {self.symbols}")

    async def initialize(self):
        for symbol in self.symbols:
            await self.set_leverage(symbol)
            await self.load_precision(symbol)
            for timeframe in self.timeframes:
                df = await self.client.fetch_candles(symbol, timeframe, limit=300)
                if df is not None and not df.empty:
                    self.data[symbol][timeframe] = df
                    logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
                else:
                    logger.warning(f"No data loaded for {symbol} {timeframe}")
        for symbol in self.symbols:
            self.cooldown[symbol] = None
            self.daily_trades[symbol] = []
            await self.sync_position(symbol)

    async def set_leverage(self, symbol):
        try:
            await self.client.exchange.fapiPrivatePostLeverage({
                'symbol': symbol.replace('/', ''),
                'leverage': self.leverage[symbol]
            })
            logger.info(f"Set leverage {self.leverage[symbol]}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")

    async def load_precision(self, symbol):
        try:
            info = await self.client.exchange.fapiPublicGetExchangeInfo()
            for m in info['symbols']:
                if m['symbol'] == symbol.replace('/', ''):
                    self.price_precision[symbol] = int(m['pricePrecision'])
                    self.quantity_precision[symbol] = int(m['quantityPrecision'])
                    break
        except Exception as e:
            logger.error(f"Failed to load precision for {symbol}: {e}")

    def process_timeframe_data(self, symbol, timeframe, kline):
        df = self.data[symbol].setdefault(timeframe, pd.DataFrame(columns=['timestamp','open','high','low','close','volume']))
        new = pd.DataFrame([{
            'timestamp': pd.to_datetime(kline['timestamp'],unit='ms'),
            'open':float(kline['open']),'high':float(kline['high']),
            'low':float(kline['low']),'close':float(kline['close']),'volume':float(kline['volume'])
        }])
        df = pd.concat([df,new]).drop_duplicates('timestamp').tail(60)
        self.data[symbol][timeframe]=df

    def process_tick(self, symbol, tick):
        for tf,df in self.data[symbol].items():
            df.iat[-1,df.columns.get_loc('close')] = float(tick)

    def get_signal_for_timeframe(self, symbol, timeframe):
        df = self.data[symbol].get(timeframe,pd.DataFrame())
        if len(df)<30: return None
        ema_s=talib.EMA(df['close'],timeperiod=self.config['indicators'][symbol].get('ema_short',12))
        ema_l=talib.EMA(df['close'],timeperiod=self.config['indicators'][symbol].get('ema_long',26))
        long=short=0
        if ema_s.iloc[-1]>ema_l.iloc[-1]: long+=2
        elif ema_s.iloc[-1]<ema_l.iloc[-1]: short+=2
        macd,signal,_=talib.MACD(df['close'],12,26,9)
        if macd.iloc[-1]>signal.iloc[-1]: long+=1
        elif macd.iloc[-1]<signal.iloc[-1]: short+=1
        rsi=talib.RSI(df['close'],timeperiod=self.config['indicators'][symbol].get('rsi',14))
        if rsi.iloc[-1]>70: short+=1
        elif rsi.iloc[-1]<30: long+=1
        adx=talib.ADX(df['high'],df['low'],df['close'],14)
        if adx.iloc[-1]>25:
            if ema_s.iloc[-1]>ema_l.iloc[-1]: long+=1.5
            else: short+=1.5
        score=long-short
        if score>=1.5: return 'long'
        if score<=-1.5: return 'short'
        return None

    def check_multi_timeframe_signal(self, symbol):
        signals,scores={},{}
        for tf in self.timeframes:
            if tf in self.data[symbol] and not self.data[symbol][tf].empty:
                sig=self.get_signal_for_timeframe(symbol,tf)
                if sig:
                    signals[tf]=sig
                    scores[tf]=self.get_signal_for_timeframe_score(symbol,tf)
        for tf in ['1h','4h','1d']:
            if tf in signals and abs(scores[tf])>=3: return signals[tf],tf
        if len(signals)>=2 and len(set(signals.values()))==1: return next(iter(signals.values())),next(iter(signals.keys()))
        return None,None

    def get_signal_for_timeframe_score(self,symbol,tf):
        df=self.data[symbol].get(tf,pd.DataFrame())
        if len(df)<30: return 0
        long=short=0
        ema_s=talib.EMA(df['close'],timeperiod=self.config['indicators'][symbol].get('ema_short',12))
        ema_l=talib.EMA(df['close'],timeperiod=self.config['indicators'][symbol].get('ema_long',26))
        if ema_s.iloc[-1]>ema_l.iloc[-1]: long+=2
        elif ema_s.iloc[-1]<ema_l.iloc[-1]: short+=2
        macd,signal,_=talib.MACD(df['close'],12,26,9)
        if macd.iloc[-1]>signal.iloc[-1]: long+=1
        elif macd.iloc[-1]<signal.iloc[-1]: short+=1
        rsi=talib.RSI(df['close'],timeperiod=self.config['indicators'][symbol].get('rsi',14))
        if rsi.iloc[-1]>70: short+=1
        elif rsi.iloc[-1]<30: long+=1
        adx=talib.ADX(df['high'],df['low'],df['close'],14)
        if adx.iloc[-1]>25:
            if ema_s.iloc[-1]>ema_l.iloc[-1]: long+=1.5
            else: short+=1.5
        return long-short

    async def sync_position(self, symbol):
        try:
            pos=await self.client.exchange.fapiPrivateV2GetPositionRisk({'symbol':symbol.replace('/','')})
            orders=await self.client.exchange.fetch_open_orders(symbol)
            active=False
            for p in pos:
                amt=float(p['positionAmt'])
                if amt!=0:
                    active=True; side='long' if amt>0 else 'short'
                    if self.position_side[symbol]!=side:
                        self.position_side[symbol]=side
                        self.quantity[symbol]=abs(amt)
                        self.entry_price[symbol]=float(p['entryPrice'])
                        self.unrealized_pnl[symbol]=float(p['unRealizedProfit'])
                        if not any(o['type'].upper() in ['STOP_MARKET','TAKE_PROFIT_MARKET'] for o in orders):
                            await self.set_sl(symbol); await self.set_tp(symbol)
                    break
            if not active and self.position_side[symbol] is not None:
                self.position_side[symbol]=None; self.entry_price[symbol]=None; self.quantity[symbol]=None; self.unrealized_pnl[symbol]=0
            for o in orders:
                if o['status']=='open' and o['type'].upper() not in ['STOP_MARKET','TAKE_PROFIT_MARKET']:
                    await self.client.exchange.cancel_order(o['id'],symbol)
        except Exception as e:
            logger.error(f"Sync error {symbol}: {e}\n{traceback.format_exc()}")

    async def open_position(self,symbol,side,price,qty,tf):
        await self.sync_position(symbol)
        if self.position_side[symbol] or await self.has_open(symbol): return
        ob=await self.client.exchange.fetch_order_book(symbol,limit=5)
        bid,ask=float(ob['bids'][0][0]),float(ob['asks'][0][0])
        lim=round(bid+(ask-bid)*0.3,self.price_precision[symbol]) if side=='long' else round(ask-(ask-bid)*0.3,self.price_precision[symbol])
        if qty<self.min_qty.get(symbol,0): return
        try:
            order=await self.client.exchange.create_limit_order(symbol,'buy' if side=='long' else 'sell',qty,lim,{'postOnly':True})
            for _ in range(3):
                await asyncio.sleep(5)
                st=await self.client.exchange.fetch_order(order['id'],symbol)
                if st['status'] in ['closed','FILLED']:
                    self.position_side[symbol]=side; self.entry_price[symbol]=float(st.get('price') or st.get('avgPrice')); self.quantity[symbol]=qty
                    await self.set_sl(symbol); await self.set_tp(symbol)
                    self.daily_trades[symbol].append(datetime.now()); self.cooldown[symbol]=datetime.now()+timedelta(minutes=self.calculate_cooldown(symbol,tf))
                    self.log_trade(symbol,'ENTRY',self.entry_price[symbol],0,'open',tf); return
                if st['status']=='CANCELED': return
            await self.client.exchange.cancel_order(order['id'],symbol)
        except ccxt.BaseError as e:
            if 'code' in str(e) and '-5022' in str(e):
                logger.warning(f"PostOnly order rejected for {symbol} ({side}) due to -5022 error, falling back to market order")
                mkt=await self.client.exchange.create_market_order(symbol,'buy' if side=='long' else 'sell',qty)
                st=await self.client.exchange.fetch_order(mkt['id'],symbol)
                if st['status'] in ['closed','FILLED']:
                    self.position_side[symbol]=side; self.entry_price[symbol]=float(st.get('price') or st.get('avgPrice')); self.quantity[symbol]=qty
                    await self.set_sl(symbol); await self.set_tp(symbol)
                    self.daily_trades[symbol].append(datetime.now()); self.cooldown[symbol]=datetime.now()+timedelta(minutes=self.calculate_cooldown(symbol,tf))
                    self.log_trade(symbol,'ENTRY',self.entry_price[symbol],0,'open',tf)
                return
            else:
                logger.error(f"Failed to place limit order for {symbol}: {e}")
                return
        mkt=await self.client.exchange.create_market_order(symbol,'buy' if side=='long' else 'sell',qty)
        st=await self.client.exchange.fetch_order(mkt['id'],symbol)
        if st['status'] in ['closed','FILLED']:
            self.position_side[symbol]=side; self.entry_price[symbol]=float(st.get('price') or st.get('avgPrice')); self.quantity[symbol]=qty
            await self.set_sl(symbol); await self.set_tp(symbol)
            self.daily_trades[symbol].append(datetime.now()); self.cooldown[symbol]=datetime.now()+timedelta(minutes=self.calculate_cooldown(symbol,tf))
            self.log_trade(symbol,'ENTRY',self.entry_price[symbol],0,'open',tf)

    def calculate_cooldown(self,symbol,tf):
        df=self.data[symbol].get(tf,pd.DataFrame())
        if len(df)<10: return 30
        vol=df['close'].pct_change().std()*100
        base={'5m':15,'15m':30,'30m':45,'1h':30,'4h':45,'1d':60}
        return min(base.get(tf,30)*(1+vol),1440)

    async def set_sl(self,symbol):
        for _ in range(2):
            if not self.entry_price[symbol] or not self.quantity[symbol]: await self.sync_position(symbol); return
            _,sl=self.calculate_tp_sl(symbol)
            if abs(sl-self.entry_price[symbol])/self.entry_price[symbol]>0.05:
                sl=round(self.entry_price[symbol]*(0.95 if self.position_side[symbol]=='long' else 1.05),self.price_precision[symbol])
            params={'stopPrice':sl,'reduceOnly':True,'timeInForce':'GTC'}
            try:
                o=await self.client.exchange.create_order(symbol,'STOP_MARKET','sell' if self.position_side[symbol]=='long' else 'buy',round(self.quantity[symbol],self.quantity_precision[symbol]),None,params)
                return
            except Exception:
                await asyncio.sleep(2)
        await self.sync_position(symbol)

    async def set_tp(self,symbol):
        for _ in range(2):
            if not self.entry_price[symbol] or not self.quantity[symbol]: await self.sync_position(symbol); return
            tp,_=self.calculate_tp_sl(symbol)
            if abs(tp-self.entry_price[symbol])/self.entry_price[symbol]>0.05:
                tp=round(self.entry_price[symbol]*(1.05 if self.position_side[symbol]=='long' else 0.95),self.price_precision[symbol])
            params={'stopPrice':tp,'reduceOnly':True,'timeInForce':'GTC'}
            try:
                await self.client.exchange.create_order(symbol,'TAKE_PROFIT_MARKET','sell' if self.position_side[symbol]=='long' else 'buy',round(self.quantity[symbol],self.quantity_precision[symbol]),None,params)
                return
            except Exception:
                await asyncio.sleep(2)
        await self.sync_position(symbol)

    def calculate_tp_sl(self,symbol):
        tp=self.entry_price[symbol]*(1+self.config['tp'].get(symbol,0.04)/self.leverage[symbol]) if self.position_side[symbol]=='long' else self.entry_price[symbol]*(1-self.config['tp'].get(symbol,0.04)/self.leverage[symbol])
        sl=self.entry_price[symbol]*(1-self.config['sl'].get(symbol,0.025)/self.leverage[symbol]) if self.position_side[symbol]=='long' else self.entry_price[symbol]*(1+self.config['sl'].get(symbol,0.025)/self.leverage[symbol])
        return round(tp,self.price_precision[symbol]),round(sl,self.price_precision[symbol])

    async def update_positions(self,symbol):
        try: await self.sync_position(symbol)
        except: pass

    async def close_position(self,symbol):
        try:
            open_orders=await self.client.exchange.fetch_open_orders(symbol)
            for o in open_orders:
                if o.get('type','').upper() in ['STOP_MARKET','TAKE_PROFIT_MARKET']:
                    await self.client.exchange.cancel_order(o['id'],symbol)
            if self.quantity[symbol] is None or self.position_side[symbol] is None:
                await self.sync_position(symbol); return
            side='sell' if self.position_side[symbol]=='long' else 'buy'
            params={'reduceOnly':True,'closePosition':True}
            order=await self.client.exchange.create_order(symbol,'MARKET',side,abs(self.quantity[symbol]),None,params)
            entry=self.entry_price[symbol]; exit_price=float(order.get('avgPrice') or order.get('price'))
            result='win' if (self.position_side[symbol]=='long' and exit_price>entry) or (self.position_side[symbol]=='short' and exit_price<entry) else 'loss'
            self.log_trade(symbol,'EXIT',entry,exit_price,result,'unknown')
            self.position_side[symbol]=None; self.entry_price[symbol]=None; self.quantity[symbol]=None; self.unrealized_pnl[symbol]=0
        except: await self.sync_position(symbol)

    def log_trade(self,symbol,trade_type,entry,exit_price,result,timeframe):
        row=[datetime.now().strftime('%Y-%m-%d %H:%M:%S'),symbol,timeframe,trade_type,entry,exit_price,((exit_price-entry)/entry*100 if trade_type=='EXIT' else 0),result]
        try:
            with open('trade_log.csv','a') as f: f.write(','.join(map(str,row))+'\n')
        except: pass

    def get_recent_trades(self,symbol=None):
        try:
            df=pd.read_csv('trade_log.csv'); df['timestamp']=pd.to_datetime(df['timestamp'])
            recent=df[df['timestamp']>=datetime.now()-timedelta(days=1)]
            return recent[recent['symbol']==symbol].to_dict('records') if symbol else recent.to_dict('records')
        except: return []

    async def calculate_qty(self,symbol,price):
        bal=await self.client.get_balance(); usdt=float(bal.get('USDT',{}).get('free',0))
        if usdt<50: return 0
        qty=round(50/float(price)*self.leverage[symbol],self.quantity_precision[symbol])
        return qty if qty>=self.min_qty.get(symbol,0) else 0

    async def has_open(self,symbol):
        orders=await self.client.exchange.fetch_open_orders(symbol)
        return any(o['type'].upper() not in ['STOP_MARKET','TAKE_PROFIT_MARKET'] for o in orders)

    async def train_mode(self, days):
        logger.info(f"Starting training mode for {days} days...")
        for symbol in self.symbols:
            self.daily_trades[symbol] = []  # Reset daily trades for simulation
            for tf in self.timeframes:
                # Fetch historical data
                since = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)
                df = await self.client.fetch_candles(symbol, tf, limit=1000)
                if df is None or df.empty:
                    logger.warning(f"No data for {symbol} {tf} in training mode")
                    continue
                self.data[symbol][tf] = df

                # Simulate trades for each candle
                for i in range(30, len(df) - 1):  # Leave last candle for exit price
                    self.data[symbol][tf] = df.iloc[:i+1]
                    sig, timeframe = self.check_multi_timeframe_signal(symbol)
                    if sig and len(self.daily_trades[symbol]) < self.max_trades:
                        price = df['close'].iloc[i]
                        qty = await self.calculate_qty(symbol, price)
                        if qty == 0:
                            continue
                        self.daily_trades[symbol].append(datetime.now())
                        self.log_trade(symbol, 'ENTRY', price, 0, 'open', tf)

                        # Simulate exit based on next candle
                        next_candle = df.iloc[i+1]
                        tp, sl = self.calculate_tp_sl(symbol)
                        exit_price = next_candle['close']
                        if sig == 'long':
                            if next_candle['high'] >= tp:
                                exit_price = tp
                                result = 'win'
                            elif next_candle['low'] <= sl:
                                exit_price = sl
                                result = 'loss'
                            else:
                                result = 'win' if exit_price > price else 'loss'
                        else:  # short
                            if next_candle['low'] <= tp:
                                exit_price = tp
                                result = 'win'
                            elif next_candle['high'] >= sl:
                                exit_price = sl
                                result = 'loss'
                            else:
                                result = 'win' if exit_price < price else 'loss'
                        self.log_trade(symbol, 'EXIT', price, exit_price, result, tf)
                        logger.info(f"Simulated trade for {symbol} {tf}: {sig}, Entry: {price}, Exit: {exit_price}, Result: {result}")
                        await asyncio.sleep(0.1)  # Avoid rate limits
        # Retrain model with simulated trades
        train_from_log()
        logger.info("Training mode completed, model retrained.")

    async def run(self):
        try:
            while True:
                for symbol in self.symbols:
                    await self.sync_position(symbol)
                    if self.cooldown[symbol] and datetime.now()<self.cooldown[symbol]: continue
                    self.daily_trades[symbol]=[t for t in self.daily_trades[symbol] if t>datetime.now()-timedelta(days=1)]
                    if len(self.daily_trades[symbol])>=self.max_trades: continue
                    if self.position_side[symbol] or await self.has_open(symbol): continue
                    sig,tf=self.check_multi_timeframe_signal(symbol)
                    if sig:
                        price=self.data[symbol][tf]['close'].iloc[-1]; qty=await self.calculate_qty(symbol,price)
                        if qty: await self.open_position(symbol,sig,price,qty,tf)
                await asyncio.sleep(60)
        finally:
            await self.client.close()
