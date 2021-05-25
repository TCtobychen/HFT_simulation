from operator import pos
from matplotlib.cm import register_cmap
import numpy as np 
import os
from numpy.core.defchararray import _startswith_dispatcher
from numpy.core.fromnumeric import _size_dispatcher, cumprod, swapaxes

from numpy.lib import index_tricks
from numpy.lib.function_base import select
from numpy.lib.histograms import histogram_bin_edges
from numpy.lib.shape_base import _hvdsplit_dispatcher
from numpy.lib.twodim_base import mask_indices

class Orderbook:
    def __init__(self, start_price, tick_size, tag):
        self.start_price = start_price
        self.tick_size = tick_size
        self.tag = tag
        self.bids = []
        self.asks = []
        self.trades = []
        self.prices = []
        self.order_id = 0
        self.show_trade_ind = 0
        self.generate_basic_liquidity()
        self.slice_size = 8
        self.time = 0
    
    def save_history(self):
        history = np.zeros((len(self.trades), 3))
        for i, item in enumerate(self.trades):
            history[i] = [item[0], item[1], item[-1]]
        np.save(self.tag+'.npy', history)

    def generate_slice(self):
        self.refresh()
        res = []
        ind, cnt = 0, 0
        while cnt < 3 and ind < len(self.bids):
            if len(self.bids[ind]) == 0:
                ind += 1
                continue
            size = 0
            for item in self.bids[ind]:
                size += item[1]
            res.append(size)
            cnt += 1
            ind += 1
        for i in range(cnt, 3):
            res.append(0)
        ind, cnt = 0, 0
        while cnt < 3 and ind < len(self.asks):
            if len(self.asks[ind]) == 0:
                ind += 1
                continue
            size = 0
            for item in self.asks[ind]:
                size += item[1]
            res.append(size)
            cnt += 1
            ind += 1
        for i in range(cnt, 3):
            res.append(0)
        current_bid, current_ask = self.quote()
        mid = (current_bid + current_ask) / 2.0
        if len(self.trades) > 0:
            res.append(self.trades[-1][0]-mid)
        else:
            res.append(0)
        res.append(mid)
        return np.array(res)

    def generate_basic_liquidity(self):
        for i in range(5):
            self.bids.append([[self.start_price - (i+1)*self.tick_size, 1, -1, 'base_liquidity']])
            self.asks.append([[self.start_price + (i+1)*self.tick_size, 1, -1, 'base_liquidity']])
    
    def show(self, only_trade = False):
        self.refresh()
        print('HISTORY TRADE:')
        for i in range(self.show_trade_ind, len(self.trades)):
            print(self.trades[i])
        if only_trade:
            self.show_trade_ind = len(self.trades)
            return 
        print('ASK SIDE: ')
        for i in range(len(self.asks)-1,-1,-1):
            print(self.asks[i])
        print('BUY SIDE: ')
        for i in range(len(self.bids)):
            print(self.bids[i])
        self.show_trade_ind = len(self.trades)
    
    def quote(self):
        self.refresh()
        try:
            assert self.bids[0][0][0]<self.asks[0][0][0]
        except:
            print(self.bids)
            print(self.asks)
            self.save_history()
            
        return self.bids[0][0][0], self.asks[0][0][0]
    
    def spread(self):
        self.refresh()
        try:
            assert self.bids[0][0][0]<self.asks[0][0][0]
        except:
            print(self.bids)
            print(self.asks)
        return (self.asks[0][0][0]-self.bids[0][0][0])/self.tick_size

    def cancel(self, order):
        for i in range(len(self.bids)):
            if len(self.bids[i]) == 0:
                continue
            if self.bids[i][0][0] != order[0]:
                continue
            try:
                self.bids[i].remove(order)
            except:
                print('buy Order not found')
                print(self.bids[i])
                print(order)
        for i in range(len(self.asks)):
            if len(self.asks[i]) == 0:
                continue
            if self.asks[i][0][0] != order[0]:
                continue
            try:
                self.asks[i].remove(order)
            except:
                print('sell Order not found')
                print(self.asks[i])
                print(order)

    def refresh(self, num_tick_away=10):
        # make sure that self.bids and self.asks start with levels with at least one order
        # and it has enough depth for traders to put order in
        if len(self.bids) == 0 or len(self.asks) == 0:
            return 
        i = 0
        while i < len(self.bids) and len(self.bids[i]) == 0:
            i += 1
        self.bids = self.bids[i:]
        for j in range(len(self.bids), num_tick_away+1):
            self.bids.append([])
        i = 0
        while i < len(self.asks) and len(self.asks[i]) == 0:
            i += 1
        self.asks = self.asks[i:]
        for j in range(len(self.asks), num_tick_away+1):
            self.asks.append([])

    def moving_average(self, num = 1200):
        if num > len(self.trades):
            num = len(self.trades)
        if len(self.prices) < len(self.trades):
            for i in range(len(self.prices), len(self.trades)):
                self.prices.append(self.trades[i][0])
        return np.mean(self.prices[-1*num:])

    def get_order_status(self, open_orders, start_ind):
        # open_orders: {'0': [price, size_left, buy/sell]}
        position_change, cash_change = 0, 0
        trading_volume = 0
        for item in self.trades[start_ind:]:
            if item[2] in open_orders.keys():
                #print(f'trade matched with id {item[2]}')
                open_orders[item[2]][1] -= item[1]
                position_change += open_orders[item[2]][2] * item[1]
                cash_change -= open_orders[item[2]][2] * item[1] * item[0]
                trading_volume += item[1] * item[0]
                if open_orders[item[2]][1] == 0:
                    del open_orders[item[2]]
        return open_orders, position_change, cash_change, len(self.trades), trading_volume


    def get_volatility(self, num):
        if num > len(self.trades):
            num = len(self.trades)
        if len(self.prices) < len(self.trades):
            for i in range(len(self.prices), len(self.trades)):
                self.prices.append(self.trades[i][0])
        return np.var(self.prices[-1*num:])

class Trader:
    def __init__(self):
        self.position = 0
        self.open_orders = {}
        self.tag = 'Trader'
        self.trade_history_ind = 0
        self.pnl = 0
        self.cash = 0
        self.trading_volume = 0

    def update_pnl(self, order_book: Orderbook):
        self.open_orders, position_change, cash_change, self.trade_history_ind, volume_change = order_book.get_order_status(self.open_orders, self.trade_history_ind)
        self.position += position_change
        self.cash += cash_change
        self.trading_volume += volume_change
        try:
            current_bid, current_ask = order_book.quote()
            settle_price = (current_bid + current_ask) / 2.0
        except:
            settle_price = order_book.trades[-1][0]
        self.pnl = self.cash + self.position * settle_price
        return self.pnl
    
    def penny_buy(self, order_book: Orderbook, size):
        order_book.refresh()
        order_book.bids = [[]] + order_book.bids
        order_book.bids[0].append([order_book.bids[1][0][0]+order_book.tick_size, size, order_book.order_id, self.tag])
        self.open_orders[order_book.order_id] = [order_book.bids[0][0][0], size, 1]
        order_book.order_id += 1
        return order_book

    def penny_sell(self, order_book: Orderbook, size):
        order_book.refresh()
        order_book.asks = [[]] + order_book.asks
        order_book.asks[0].append([order_book.asks[1][0][0]-order_book.tick_size, size, order_book.order_id, self.tag])
        self.open_orders[order_book.order_id] = [order_book.asks[0][0][0], size, -1]
        order_book.order_id += 1
        return order_book

    def join_buy(self, order_book: Orderbook, num_tick_away, size):
        num_tick_away = int(num_tick_away)
        order_book.refresh(num_tick_away)
        current_bid, current_ask = order_book.quote()
        #print(order_book.bids, len(order_book.bids), num_tick_away)
        order_book.bids[num_tick_away].append([current_bid - num_tick_away * order_book.tick_size, size, order_book.order_id, self.tag])
        self.open_orders[order_book.order_id] = [current_bid - num_tick_away * order_book.tick_size, size, 1]
        order_book.order_id += 1
        return order_book

    def join_sell(self, order_book: Orderbook, num_tick_away, size):
        num_tick_away = int(num_tick_away)
        order_book.refresh(num_tick_away)
        current_bid, current_ask = order_book.quote()
        order_book.asks[num_tick_away].append([current_ask + num_tick_away * order_book.tick_size, size, order_book.order_id, self.tag])
        self.open_orders[order_book.order_id] = [current_ask + num_tick_away * order_book.tick_size, size, -1]
        order_book.order_id += 1
        return order_book

    def market_buy(self, order_book: Orderbook, size):
        # takes care of change of position, cash
        order_book.refresh()
        size_left = size
        sweep_ind_i, sweep_ind_j = 0, 0
        for i, item in enumerate(order_book.asks):
            sweep_ind_j = 0
            for j, sell_order in enumerate(item):
                if sell_order[1] <= size_left:
                    size_left -= sell_order[1]
                    self.cash -= sell_order[1] * sell_order[0]
                    self.position += sell_order[1]
                    self.trading_volume += sell_order[1] * sell_order[0]
                    sweep_ind_j = j+1
                    order_book.trades.append(sell_order+[self.tag, 1, order_book.time])
                    if size_left == 0:
                        break
                else:
                    order_book.asks[i][j][1] -= size_left
                    order_book.trades.append([sell_order[0], size_left, sell_order[2],sell_order[3], self.tag, 1, order_book.time])
                    self.cash -= size_left * sell_order[0]
                    self.position += size_left
                    self.trading_volume += size_left * sell_order[0]
                    size_left = 0
                    break
            if size_left == 0:
                break
            sweep_ind_i = i+1
        order_book.asks = order_book.asks[sweep_ind_i:]
        if len(order_book.asks) == 0:
            order_book.show()
            print('NO more sell orders to be hit, system is fucked up!!!!')
            return order_book
        order_book.asks[0] = order_book.asks[0][sweep_ind_j:]
        return order_book
    
    def market_sell(self, order_book: Orderbook, size):
        order_book.refresh()
        size_left = size
        sweep_ind_i, sweep_ind_j = 0, 0
        for i, item in enumerate(order_book.bids):
            sweep_ind_j = 0
            for j, buy_order in enumerate(item):
                if buy_order[1] <= size_left:
                    size_left -= buy_order[1]
                    sweep_ind_j = j+1
                    order_book.trades.append(buy_order+[self.tag, -1, order_book.time])
                    self.cash += buy_order[1] * buy_order[0]
                    self.position -= buy_order[1]
                    self.trading_volume += buy_order[1] * buy_order[0]
                    if size_left == 0:
                        break
                else:
                    order_book.bids[i][j][1] -= size_left
                    order_book.trades.append([buy_order[0], size_left, buy_order[2], buy_order[3], self.tag, -1, order_book.time])
                    self.cash += size_left * buy_order[0]
                    self.position -= size_left
                    self.trading_volume += size_left * buy_order[0]
                    size_left = 0
                    break
            if size_left == 0:
                break
            sweep_ind_i = i+1
        order_book.bids = order_book.bids[sweep_ind_i:]
        if len(order_book.bids) == 0:
            order_book.show()
            print('NO more buy orders to be hit, system is fucked up!!!!')
            return order_book
        order_book.bids[0] = order_book.bids[0][sweep_ind_j:]
        return order_book

class MarketMaker(Trader):
    def __init__(self, position_limit, base_volume, base_volatility, start_cash, cnt = 0):
        Trader.__init__(self)
        self.position_limit = position_limit
        self.base_volume = base_volume
        self.base_volatility = base_volatility
        self.position = 0
        self.cash = start_cash
        self.open_orders = {}
        self.trade_history_ind = 0
        self.tag = f'MarketMaker_{cnt+1}'
    
    def show(self):
        print(f'{self.tag}: {self.position}, {self.cash}, {self.open_orders}')

    def get_open_order_position(self):
        res = 0
        for order in self.open_orders.values():
            res += order[1]
        return res

    def cancel_outside(self, order_book: Orderbook):
        min_buy, max_sell = 1e9, -1e9
        for key, value in self.open_orders.items():
            if value[2] == 1:
                min_buy = min(value[0], min_buy)
            else:
                max_sell = max(value[0], max_sell)
        buy_cancel, sell_cancel = [], []
        cancel_keys = []
        for key, value in self.open_orders.items():
            if value[2] == 1 and value[0] == min_buy:
                buy_cancel.append(value[:2]+[key, self.tag])
                cancel_keys.append(key)
            if value[2] == -1 and value[0] == max_sell:
                sell_cancel.append(value[:2]+[key, self.tag])
                cancel_keys.append(key)
        for key in cancel_keys:
            del self.open_orders[key]
        for buy_order in buy_cancel:
            #print(f'Cancel buy order {buy_order}')
            order_book.cancel(buy_order)
        for sell_order in sell_cancel:
            #print(f'Cancel sell order {sell_order}')
            order_book.cancel(sell_order)
        return order_book
        
    
    def run(self, order_book: Orderbook):
        #current_vol = order_book.get_volatility()
        current_vol = self.base_volatility
        # update all trades to market maker
        #print(f'Market maker current status: {self.position}')
        self.open_orders, position_change, cash_change, self.trade_history_ind, volume_change = order_book.get_order_status(self.open_orders, self.trade_history_ind)
        self.position += position_change
        self.cash += cash_change
        self.trading_volume += volume_change
        #print(f'pc is {position_change}, cc is {cash_change}')
        #print(f'Market maker current status: {self.position}, {self.cash}')
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        self.pnl = self.cash + self.position * current_mid

        # control risk for market makers
        current_position_limit = min(self.position_limit * self.base_volatility / current_vol, self.position_limit)
        if current_position_limit < abs(self.position):
            if self.position > 0:
                order_book = self.market_sell(order_book, self.position - current_position_limit)
            else:
                order_book = self.market_buy(order_book, abs(self.position) - current_position_limit)
        self.open_orders, position_change, cash_change, self.trade_history_ind, volume_change = order_book.get_order_status(self.open_orders, self.trade_history_ind)
        self.position += position_change
        self.cash += cash_change
        self.trading_volume += volume_change
        can_buy = self.position < current_position_limit
        can_sell = self.position > -1*current_position_limit 
        # make market
        position_left = current_position_limit - abs(self.position)
        if self.get_open_order_position() > position_left:
            order_book = self.cancel_outside(order_book)
        if self.get_open_order_position() > position_left * 3:
            order_book.time += 1
            return order_book
        #print(f'Market maker current status: {position_left}, {current_position_limit}, {self.position}')
        current_bid, current_ask = order_book.quote()
        #print(current_bid, current_ask, order_book.tick_size)
        if current_ask - current_bid == 2  * order_book.tick_size:
            if self.position > 0:
                if can_sell:
                    order_book = self.penny_sell(order_book, position_left//10+1)
            elif can_buy:
                order_book = self.penny_buy(order_book, position_left//10+1)
        if current_ask - current_bid > 2 * order_book.tick_size:
            print(f'{self.tag} pennying with current bid ask {current_bid}, {current_ask}. Order id now is {order_book.order_id}')
            if can_buy:
                order_book = self.penny_buy(order_book, position_left//10+1)
            if can_sell:
                order_book = self.penny_sell(order_book, position_left//10+1)
        for i in range(6):
            #order_book.show()
            if can_buy:
                order_book = self.join_buy(order_book, i, position_left*i//10+1)
            if can_sell:
                order_book = self.join_sell(order_book, i, position_left*i//10+1)
        order_book.time += 1
        return order_book
    
class HFT(Trader):
    def __init__(self, position_limit, start_cash, cnt = 0, model=None, edge=0):
        Trader.__init__(self)
        self.position_limit = position_limit
        self.position = 0
        self.trade_history_ind = 0
        self.cash = start_cash
        self.model = model
        self.edge = edge
        self.tag = f'HFT_{cnt+1}'
        self.enter_price = None
        self.enter_time = 0

    def market_buy(self, order_book: Orderbook, size):
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        # takes care of change of position, cash
        order_book.refresh()
        size_left = size
        sweep_ind_i, sweep_ind_j = 0, 0
        for i, item in enumerate(order_book.asks):
            sweep_ind_j = 0
            for j, sell_order in enumerate(item):
                if sell_order[1] <= size_left:
                    size_left -= sell_order[1]
                    self.cash -= sell_order[1] * current_mid
                    self.position += sell_order[1]
                    self.trading_volume += sell_order[1] * current_mid
                    sweep_ind_j = j+1
                    order_book.trades.append(sell_order+[self.tag, 1, order_book.time])
                    if size_left == 0:
                        break
                else:
                    order_book.asks[i][j][1] -= size_left
                    order_book.trades.append([sell_order[0], size_left, sell_order[2],sell_order[3], self.tag, 1, order_book.time])
                    self.cash -= size_left * current_mid
                    self.position += size_left
                    self.trading_volume += size_left * current_mid
                    size_left = 0
                    break
            if size_left == 0:
                break
            sweep_ind_i = i+1
        order_book.asks = order_book.asks[sweep_ind_i:]
        if len(order_book.asks) == 0:
            order_book.show()
            print('NO more sell orders to be hit, system is fucked up!!!!')
            return order_book
        order_book.asks[0] = order_book.asks[0][sweep_ind_j:]
        return order_book
    
    def market_sell(self, order_book: Orderbook, size):
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        order_book.refresh()
        size_left = size
        sweep_ind_i, sweep_ind_j = 0, 0
        for i, item in enumerate(order_book.bids):
            sweep_ind_j = 0
            for j, buy_order in enumerate(item):
                if buy_order[1] <= size_left:
                    size_left -= buy_order[1]
                    sweep_ind_j = j+1
                    order_book.trades.append(buy_order+[self.tag, -1, order_book.time])
                    self.cash += buy_order[1] * current_mid
                    self.position -= buy_order[1]
                    self.trading_volume += buy_order[1] * current_mid
                    if size_left == 0:
                        break
                else:
                    order_book.bids[i][j][1] -= size_left
                    order_book.trades.append([buy_order[0], size_left, buy_order[2], buy_order[3], self.tag, -1, order_book.time])
                    self.cash += size_left * current_mid
                    self.position -= size_left
                    self.trading_volume += size_left * current_mid
                    size_left = 0
                    break
            if size_left == 0:
                break
            sweep_ind_i = i+1
        order_book.bids = order_book.bids[sweep_ind_i:]
        if len(order_book.bids) == 0:
            order_book.show()
            print('NO more buy orders to be hit, system is fucked up!!!!')
            return order_book
        order_book.bids[0] = order_book.bids[0][sweep_ind_j:]
        return order_book

    def get_prediction_last_trade(self, order_book: Orderbook):
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        last_price = order_book.trades[-1][0]
        diff = last_price - current_mid
        return diff * np.random.random()*3

    def get_prediction(self, order_book: Orderbook):
        if self.model is None:
            return np.random.random() * 5 - 2.5
        features = order_book.generate_slice()
        return self.model.predict(features[:-1])

    def run(self, order_book: Orderbook):
        if self.position != 0 and order_book.time > self.enter_time + 60:
            current_bid, current_ask = order_book.quote()
            current_mid = (current_bid + current_ask) / 2
            print(f'{self.tag} flat {self.position} at time {order_book.time} with enter time {self.enter_time}, enter_price {self.enter_price}, current quote {current_bid}, {current_ask}')
            if self.position > 0: 
                order_book = self.market_sell(order_book, self.position)
            else:
                order_book = self.market_buy(order_book, abs(self.position))
            order_book.time += 1
            return order_book
        pred = self.get_prediction(order_book)
        #pred = 0
        #print(f'{self.tag} Trading with prediction: {pred}')
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        if current_mid + pred > current_ask + self.edge:
            self.enter_price = current_ask
            self.enter_time = order_book.time
            ask_size = order_book.generate_slice()[3]
            size = min(ask_size, self.position_limit-abs(self.position))
            #print(f'{self.tag} buying with size {size}')
            order_book = self.market_buy(order_book, size)
        if current_mid + pred < current_bid- self.edge:
            self.enter_price = current_bid
            self.enter_time = order_book.time
            bid_size = order_book.generate_slice()[0]
            size = min(bid_size, self.position_limit-abs(self.position))
            #print(f'{self.tag} selling with size {size}')
            order_book = self.market_sell(order_book, size)
        #print(f'HFT status after trading: position is {self.position}, cash is {self.cash}')
        order_book.time += 1
        return order_book

class Spoofer(Trader):
    def __init__(self, max_volume, fake_volume, flat_time = 100, fake_level = 1, cnt = 0):
        Trader.__init__(self)
        self.max_volume = max_volume
        self.fake_volume = fake_volume
        self.flat_time = flat_time
        self.fake_level = fake_level
        self.tag = f'Spoofer_{cnt+1}'
        self.have_info = False
        self.fake_orders = {}
        self.spoof_time = 0
        self.spoof_price = 0
        self.info = 0

    def set_info(self, info, time):
        self.have_info = True
        self.spoof_time = time
        self.info = info

    def run(self, order_book: Orderbook):
        #print(f'{self.tag} called with status {self.have_info}, {self.info}, {self.position}, {self.fake_orders}')
        if not self.have_info:
            return order_book
        # Put fake large orders on one side
        # Cancel next turn
        self.open_orders, position_change, cash_change, self.trade_history_ind, volume_change = order_book.get_order_status(self.open_orders, self.trade_history_ind)
        self.position += position_change
        self.cash += cash_change
        self.trading_volume += volume_change
        for key in list(self.fake_orders.keys()):
            if key in self.open_orders:
                self.fake_orders[key] = self.open_orders[key]
            else:
                del self.fake_orders[key]
        if len(self.fake_orders.keys()) > 0:
            #print(f'{self.tag} cancel {len(self.fake_orders.keys())} spoofing orders with status {self.position}, {self.info}')
            for key in list(self.fake_orders.keys()):
                #print(self.fake_orders)
                order_book.cancel(self.fake_orders[key][:2]+[key, self.tag])
                del self.fake_orders[key]
                del self.open_orders[key]
        if order_book.time > self.spoof_time + self.flat_time:
            #print(f'{self.tag} time due with {order_book.time} and {self.spoof_time}')
            current_bid, current_ask = order_book.quote()
            current_mid = (current_bid + current_ask) / 2.0
            self.have_info = False
            self.info = 0
            if self.position > 0:
                #print(f'{self.tag} flat {self.position} at price {current_mid}, {current_bid} at time {order_book.time} with enter price {self.spoof_price}')
                order_book = self.market_sell(order_book, self.position)
            if self.position < 0:
                #print(f'{self.tag} flat {self.position} at price {current_mid}, {current_ask} at time {order_book.time} with enter price {self.spoof_price}')
                order_book = self.market_buy(order_book, abs(self.position))
            order_book.time += 1
            for key in list(self.open_orders.keys()):
                order_book.cancel(self.open_orders[key][:2]+[key, self.tag])
                del self.open_orders[key]
                if key in self.fake_orders.keys():
                    del self.fake_orders[key]
            return order_book
        if self.info > 0:
            self.info = 0
            current_bid, current_ask = order_book.quote()
            current_mid = (current_bid + current_ask) / 2.0
            self.spoof_price = current_mid
            #print(f'{self.tag} spoof sell at price {current_mid}, at time {order_book.time}')
            self.spoof_time = order_book.time
            self.fake_orders[order_book.order_id] = 1
            order_book = self.join_sell(order_book, self.fake_level, self.fake_volume)
            order_book = self.join_buy(order_book, 0, self.max_volume)
            order_book.time += 1
            return order_book
        if self.info < 0:
            self.info = 0
            current_bid, current_ask = order_book.quote()
            current_mid = (current_bid + current_ask) / 2.0
            self.spoof_price = current_mid
            #print(f'{self.tag} spoof buy at price {current_mid}, at time {order_book.time}')
            self.spoof_time = order_book.time
            self.fake_orders[order_book.order_id] = 1
            order_book = self.join_buy(order_book, self.fake_level, self.fake_volume)
            order_book = self.join_sell(order_book, 0, self.max_volume)
            order_book.time += 1
            return order_book
        return order_book


class RandomRetailer(Trader):
    def __init__(self, max_volume, hit_ratio, cnt = 0):
        Trader.__init__(self)
        self.max_volume = max_volume
        self.hit_ratio = hit_ratio
        self.open_orders = {}
        self.tag = f'RandomRetailer_{cnt+1}'

    def buy(self, order_book: Orderbook, volume):
        order_book = self.market_buy(order_book, int(volume*self.hit_ratio)+1)
        if self.hit_ratio != 1:
            order_book = self.join_buy(order_book, int(np.random.random()*3),int(volume*self.hit_ratio)+1)
        return order_book
    
    def sell(self, order_book: Orderbook, volume):
        order_book = self.market_sell(order_book, int(volume*self.hit_ratio)+1)
        if self.hit_ratio != 1:
            order_book = self.join_sell(order_book, int(np.random.random()*3),int(volume*self.hit_ratio)+1)
        return order_book
    
    def run(self, order_book: Orderbook):
        volume = int(self.max_volume * np.random.random()) + 1
        if np.random.random() > 0.5:
            #print(f'Random Retailer buy with volume {volume}')
            order_book = self.buy(order_book, volume)
        else:
            #print(f'Random Retailer sell with volume {volume}')
            order_book = self.sell(order_book, volume)
        order_book.time += 1
        return order_book

class MARetailer(MarketMaker):
    def __init__(self, max_volume, tick_away, ma_num, ma_ratio, divides = 3, cnt=0):
        Trader.__init__(self)
        self.max_volume = max_volume
        self.tick_away = tick_away
        self.ma_num = ma_num
        self.ma_ratio = ma_ratio
        self.divides = divides
        self.left = 0
        self.trade_history_ind = 0
        self.open_orders = {}
        self.tag = f'MARetailer_{cnt+1}'
        self.seen_prices = []
    
    def run(self, order_book: Orderbook):
        current_bid, current_ask = order_book.quote()
        try:
            last_price = order_book.trades[-1][0]
        except:
            last_price = (current_bid+current_ask)/2
        self.seen_prices.append(last_price)
        moving_average = np.mean(self.seen_prices) if len(self.seen_prices)<self.ma_num else np.mean(self.seen_prices[-1*self.ma_num:])
        if last_price < moving_average*(1-self.ma_ratio):
            self.left = self.divides
        if last_price > moving_average*(1+self.ma_ratio):
            self.left = -1 * self.divides
        if self.left > 0:
            #print(f'{self.tag} found price too low compared to ma {moving_average}, pennying buy at last price {last_price}')
            order_book = self.market_buy(order_book, int(self.max_volume*0.1)+1)
            if current_ask - current_bid >= 2 * order_book.tick_size:
                order_book = self.penny_buy(order_book, int(self.max_volume*0.1)+1)
            else:
                order_book = self.join_buy(order_book, 0, int(self.max_volume*0.1)+1)
            self.left -= 1
        if self.left < 0:
            #print(f'{self.tag} found price too high compared to ma {moving_average}, pennying sell at last price {last_price}')
            order_book = self.market_sell(order_book, int(self.max_volume*0.1)+1)
            if current_ask - current_bid >= 2 * order_book.tick_size:
                order_book = self.penny_sell(order_book, int(self.max_volume*0.1)+1)
            else:
                order_book = self.join_sell(order_book, 0, int(self.max_volume*0.1)+1)
            self.left += 1
        self.open_orders, position_change, cash_change, self.trade_history_ind, volume_change = order_book.get_order_status(self.open_orders, self.trade_history_ind)
        self.position += position_change
        self.cash += cash_change
        self.trading_volume += volume_change
        current_bid, current_ask = order_book.quote()
        order_book = self.cancel_outside(order_book)
        #print(f'{self.tag} join with volume {self.max_volume} at last price {last_price}, with current bid and ask {current_bid}, {current_ask}')
        if last_price-current_bid > self.tick_away * order_book.tick_size:
            order_book = self.penny_buy(order_book, self.max_volume)
        else:
            order_book = self.join_buy(order_book, self.tick_away-(last_price-current_bid)/order_book.tick_size,self.max_volume)
        if current_ask - last_price > self.tick_away * order_book.tick_size:
            order_book = self.penny_sell(order_book, self.max_volume)
        else:
            order_book = self.join_sell(order_book, self.tick_away-(current_ask-last_price)/order_book.tick_size,self.max_volume)
        order_book.time += 1
        return order_book

class MomentumRetailer(MarketMaker):
    def __init__(self, max_volume, divides = 5, cnt=0):
        Trader.__init__(self)
        self.max_volume = max_volume
        self.direction = 1-2*int(np.random.random()>0.5)
        self.divides = divides
        self.left = int(np.random.random()*divides+1) * self.direction
        self.trade_history_ind = 0
        self.open_orders = {}
        self.tag = f'MomentumRetailer_{cnt+1}'
        self.seen_prices = []
    
    def run(self, order_book: Orderbook):
        current_bid, current_ask = order_book.quote()
        #print(f'{self.tag} get called with {self.left} times left and direction {self.direction}')
        if self.left > 0:
            #print(f'{self.tag} bringing momentum with {self.left} times left and direction {self.direction}')
            order_book = self.market_buy(order_book, int(self.max_volume*np.random.random())+1)
            if current_ask - current_bid >= 2 * order_book.tick_size:
                order_book = self.penny_buy(order_book, int(self.max_volume*np.random.random())+1)
            else:
                order_book = self.join_buy(order_book, 0, int(self.max_volume*np.random.random())+1)
            order_book = self.market_buy(order_book, int(self.max_volume*np.random.random())+1)
            self.left -= 1
            #print(f'{self.tag} done momentum with {self.left} times left and direction {self.direction}')
        if self.left < 0:
            #print(f'{self.tag} bringing momentum with {self.left} times left and direction {self.direction}')
            order_book = self.market_sell(order_book, int(self.max_volume*np.random.random())+1)
            if current_ask - current_bid >= 2 * order_book.tick_size:
                order_book = self.penny_sell(order_book, int(self.max_volume*np.random.random())+1)
            else:
                order_book = self.join_sell(order_book, 0, int(self.max_volume*np.random.random())+1)
            order_book = self.market_sell(order_book, int(self.max_volume*np.random.random())+1)
            self.left += 1
            #print(f'{self.tag} done momentum with {self.left} times left and direction {self.direction}')
        if self.left == 0:
            self.direction = 1-2*int(np.random.random()>0.5)
            self.left = int(np.random.random()*self.divides+1) * self.direction
            #print(f'{self.tag} reset momentum with {self.left} times left and direction {self.direction}')
        self.open_orders, position_change, cash_change, self.trade_history_ind, volume_change = order_book.get_order_status(self.open_orders, self.trade_history_ind)
        self.position += position_change
        self.cash += cash_change
        self.trading_volume += volume_change
        #print(f'{self.tag} finished with {self.left} times left and direction {self.direction}')
        order_book.time += 1
        return order_book

class Model:
    def __init__(self, parameters=None):
        if parameters is not None:
            self.parameters = parameters
        
    def predict(self, features):
        return np.sum(np.multiply(self.parameters, features))

    def set_parameter(self, parameters):
        self.parameters = parameters