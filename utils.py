from operator import pos
from matplotlib.cm import register_cmap
import numpy as np 
import os
from numpy.core.defchararray import _startswith_dispatcher
from numpy.core.fromnumeric import _size_dispatcher, cumprod

from numpy.lib import index_tricks
from numpy.lib.histograms import histogram_bin_edges

class Orderbook:
    def __init__(self, start_price, tick_size):
        self.start_price = start_price
        self.tick_size = tick_size
        self.bids = []
        self.asks = []
        self.trades = []
        self.prices = []
        self.order_id = 0
        self.show_trade_ind = 0
        self.generate_basic_liquidity()

    def generate_basic_liquidity(self):
        for i in range(5):
            self.bids.append([[self.start_price - (i+1)*self.tick_size, 1, -1]])
            self.asks.append([[self.start_price + (i+1)*self.tick_size, 1, -1]])
    
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
        return self.bids[0][0][0], self.asks[0][0][0]

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
        while len(self.bids[i]) == 0:
            i += 1
        self.bids = self.bids[i:]
        for j in range(len(self.bids), num_tick_away):
            self.bids.append([])
        i = 0
        while len(self.asks[i]) == 0:
            i += 1
        self.asks = self.asks[i:]
        for j in range(len(self.asks), num_tick_away):
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
        for item in self.trades[start_ind:]:
            if item[2] in open_orders.keys():
                print(f'trade matched with id {item[2]}')
                open_orders[item[2]][1] -= item[1]
                position_change += open_orders[item[2]][2] * item[1]
                cash_change -= open_orders[item[2]][2] * item[1] * item[0]
                if open_orders[item[2]][1] == 0:
                    del open_orders[item[2]]
        return open_orders, position_change, cash_change, len(self.trades)


    def get_volatility(self, num):
        if num > len(self.trades):
            num = len(self.trades)
        if len(self.prices) < len(self.trades):
            for i in range(len(self.prices), len(self.trades)):
                self.prices.append(self.trades[i][0])
        return np.var(self.prices[-1*num:])

class Trader:
    def __init__(self, position_limit):
        self.position_limit = position_limit
        self.position = 0
        self.open_orders = {}
    
    def penny_buy(self, order_book: Orderbook, size):
        order_book.refresh()
        order_book.bids = [[]] + order_book.bids
        order_book.bids[0].append([order_book.bids[1][0][0]+order_book.tick_size, size, order_book.order_id])
        self.open_orders[order_book.order_id] = [order_book.bids[0][0][0], size, 1]
        order_book.order_id += 1
        return order_book

    def penny_sell(self, order_book: Orderbook, size):
        order_book.refresh()
        order_book.asks = [[]] + order_book.asks
        order_book.asks[0].append([order_book.asks[1][0][0]-order_book.tick_size, size, order_book.order_id])
        self.open_orders[order_book.order_id] = [order_book.asks[0][0][0], size, -1]
        order_book.order_id += 1
        return order_book

    def join_buy(self, order_book: Orderbook, num_tick_away, size):
        order_book.refresh(num_tick_away)
        current_bid, current_ask = order_book.quote()
        #print(order_book.bids, len(order_book.bids), num_tick_away)
        if len(order_book.bids) == num_tick_away:
            print('Refresh is fucked up')
            order_book.show()
        order_book.bids[num_tick_away].append([current_bid - num_tick_away * order_book.tick_size, size, order_book.order_id])
        self.open_orders[order_book.order_id] = [current_bid - num_tick_away * order_book.tick_size, size, 1]
        order_book.order_id += 1
        return order_book

    def join_sell(self, order_book: Orderbook, num_tick_away, size):
        order_book.refresh(num_tick_away)
        current_bid, current_ask = order_book.quote()
        order_book.asks[num_tick_away].append([current_ask + num_tick_away * order_book.tick_size, size, order_book.order_id])
        self.open_orders[order_book.order_id] = [current_ask + num_tick_away * order_book.tick_size, size, -1]
        order_book.order_id += 1
        return order_book

    def market_buy(self, order_book: Orderbook, size):
        order_book.refresh()
        size_left = size
        sweep_ind_i, sweep_ind_j = 0, 0
        for i, item in enumerate(order_book.asks):
            for j, sell_order in enumerate(item):
                if sell_order[1] <= size_left:
                    size_left -= sell_order[1]
                    sweep_ind_j = j+1
                    order_book.trades.append(sell_order)
                    if size_left == 0:
                        break
                else:
                    order_book.asks[i][j][1] -= size_left
                    order_book.trades.append([sell_order[0], size_left, sell_order[2]])
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
            for j, buy_order in enumerate(item):
                if buy_order[1] <= size_left:
                    size_left -= buy_order[1]
                    sweep_ind_j = j+1
                    order_book.trades.append(buy_order)
                    if size_left == 0:
                        break
                else:
                    order_book.bids[i][j][1] -= size_left
                    order_book.trades.append([buy_order[0], size_left, buy_order[2]])
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
    def __init__(self, position_limit, base_volume, base_volatility, start_cash):
        self.position_limit = position_limit
        self.base_volume = base_volume
        self.base_volatility = base_volatility
        self.position = 0
        self.cash = start_cash
        self.open_orders = {}
        self.trade_history_ind = 0

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
                buy_cancel.append(value[:2]+[key])
                cancel_keys.append(key)
            if value[2] == -1 and value[0] == max_sell:
                sell_cancel.append(value[:2]+[key])
                cancel_keys.append(key)
        for key in cancel_keys:
            del self.open_orders[key]
        for buy_order in buy_cancel:
            print(f'Cancel buy order {buy_order}')
            order_book.cancel(buy_order)
        for sell_order in sell_cancel:
            print(f'Cancel sell order {sell_order}')
            order_book.cancel(sell_order)
        return order_book
        
    
    def run(self, order_book: Orderbook):
        #current_vol = order_book.get_volatility()
        current_vol = self.base_volatility
        # update all trades to market maker
        print(f'Market maker current status: {self.position}')
        self.open_orders, position_change, cash_change, self.trade_history_ind = order_book.get_order_status(self.open_orders, self.trade_history_ind)
        self.position += position_change
        self.cash += cash_change
        print(f'pc is {position_change}, cc is {cash_change}')
        print(f'Market maker current status: {self.position}, {self.cash}')
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        self.pnl = self.cash + self.position * current_mid

        # control risk for market makers
        current_position_limit = min(self.position_limit * self.base_volatility / current_vol, self.position_limit)
        if current_position_limit < abs(self.position):
            if self.position > 0:
                order_book = self.market_sell(order_book, self.position-current_position_limit)
                self.position = current_position_limit
            else:
                order_book = self.market_buy(order_book, abs(self.position)-current_position_limit)
                self.position = -1 * current_position_limit

        # make market
        position_left = current_position_limit - abs(self.position)
        if self.get_open_order_position() > position_left * 3:
            order_book = self.cancel_outside(order_book)
        if position_left <= 10 or self.get_open_order_position() > position_left * 5:
            return order_book
        print(f'Market maker current status: {position_left}, {current_position_limit}, {self.position}')
        current_bid, current_ask = order_book.quote()
        #print(current_bid, current_ask, order_book.tick_size)
        if current_ask - current_bid > 2 * order_book.tick_size:
            print(f'Market maker pennying')
            order_book = self.penny_buy(order_book, position_left//10)
            order_book = self.penny_sell(order_book, position_left//10)
        for i in range(6):
            #order_book.show()
            order_book = self.join_buy(order_book, i, position_left//10)
            order_book = self.join_sell(order_book, i, position_left//10)
        '''
        if current_ask - current_bid > 2 * order_book.tick_size:
            order_book.bids = [[current_bid + order_book.tick_size, position_left//2, order_book.order_id]] + order_book.bids
            order_book.order_id += 1
            order_book.asks = [[current_ask - order_book.tick_size, position_left//2, order_book.order_id]] + order_book.asks
            order_book.order_id += 1
        order_book_ind = 1
        for i in range(5):
            join_price = current_bid - i * order_book.tick_size
            if order_book.bids[order_book_ind][0] == join_price and order_book.bids[order_book_ind][2] in self.open_orders.keys():
                order_book.bids[order_book_ind][1] = max(order_book.bids[order_book_ind][1], position_left//2)
                self.open_orders[order_book.bids[order_book_ind][2]][1] = max(order_book.bids[order_book_ind][1], position_left//2)
                order_book_ind += 1
                continue
            if order_book.bids[order_book_ind][0] != join_price:
                order_book.bids = order_book.bids[:order_book_ind] + [[join_price, position_left//2, order_book.order_id]] + order_book.bids[order_book_ind:]
                self.open_orders[str(order_book.order_id)] = [join_price, position_left//2, 1]
                order_book.order_id += 1
                continue
        order_book_ind = 1
        for i in range(5):
            join_price = current_ask + i * order_book.tick_size
            if order_book.asks[order_book_ind][0] == join_price and order_book.asks[order_book_ind][2] in self.open_orders.keys():
                order_book.asks[order_book_ind][1] = max(order_book.asks[order_book_ind][1], position_left//2)
                self.open_orders[order_book.asks[order_book_ind][2]][1] = max(order_book.asks[order_book_ind][1], position_left//2)
                order_book_ind += 1
                continue
            if order_book.asks[order_book_ind][0] != join_price:
                order_book.asks = order_book.asks[:order_book_ind] + [[join_price, position_left//2, order_book.order_id]] + order_book.asks[order_book_ind:]
                self.open_orders[str(order_book.order_id)] = [join_price, position_left//2, -1]
                order_book.order_id += 1
                continue
        '''
        return order_book
    
class HFT(Trader):
    def __init__(self, position_limit, start_cash, model=None):
        self.position_limit = position_limit
        self.position = 0
        self.trade_history_ind = 0
        self.cash = start_cash
        self.model = model
    
    def get_prediction(self, order_book: Orderbook):
        if self.model is None:
            return np.random.random() * 5 - 2.5
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        last_price = order_book.trades[-1][0]
        features = []
        for i in range(5):
            features.append(order_book.bids[i][1])
        for i in range(5):
            features.append(order_book.asks[i][1])
        features.append(last_price-current_mid)
        return self.model.predict(features)

    def run(self, order_book: Orderbook):
        pred = self.get_prediction(order_book)
        print(f'HFT Trading with prediction: {pred}')
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        if current_mid + pred > current_ask:
            size = order_book.asks[0][0][1]
            print(f'HFT buying with size {size}')
            order_book = self.market_buy(order_book, size)
            self.position += size
            self.cash -= current_ask * size
        if current_mid + pred < current_bid:
            size = order_book.bids[0][0][1]
            print(f'HFT selling with size {size}')
            order_book = self.market_sell(order_book, size)
            self.position -= size
            self.cash += current_bid * size
        print(f'HFT status after trading: position is {self.position}, cash is {self.cash}')
        return order_book

class Retailer(Trader):
    def buy(self, order_book: Orderbook, volume):
        if np.random.random() < self.hit_ratio:
            order_book = self.market_buy(order_book, volume)
        else:
            order_book = self.join_buy(order_book, int(np.random.random()*5),volume)
        return order_book
    
    def sell(self, order_book: Orderbook, volume):
        if np.random.random() < self.hit_ratio:
            order_book = self.market_sell(order_book, volume)
        else:
            order_book = self.join_sell(order_book, int(np.random.random()*5),volume)
        return order_book



class RandomRetailer(Retailer):
    def __init__(self, max_volume, hit_ratio):
        self.max_volume = max_volume
        self.hit_ratio = hit_ratio
        self.open_orders = {}
    
    def run(self, order_book: Orderbook):
        if np.random.random() > 0.5:
            order_book = self.buy(order_book, int(self.max_volume * np.random.random())+1)
        else:
            order_book = self.sell(order_book, int(self.max_volume * np.random.random())+1)
        return order_book

class MARetailer(Retailer):
    def __init__(self, max_volume, hit_ratio, trade_ma_num, trade_ma_ratio):
        self.max_volume = max_volume
        self.hit_ratio = hit_ratio
        self.trade_ma_num = trade_ma_num
        self.trade_ma_ratio = trade_ma_ratio
    
    def run(self, order_book: Orderbook):
        ma = order_book.moving_average(num=self.trade_ma_num)
        current_bid, current_ask = order_book.quote()
        current_mid = (current_bid + current_ask) / 2
        if (ma-current_mid)/ma > self.trade_ma_ratio:
            self.market_buy


'''
class Trader:
    def __init__(self, position_limit):
        self.position_limit = position_limit
        self.position = 0
        self.open_orders = {}

    def buy(self, order_book, price, size):
        size_left = size
        sweep_ind = 0
        for i, item in enumerate(order_book.asks):
            if price >= item[0] and item[2] not in self.open_orders.keys():
                if size_left >= item[1]:
                    sweep_ind = i+1
                    size_left -= item[1]
                    order_book.trades.append(item)
                else:
                    order_book.asks[i][1] -= size_left
                    order_book.trades.append([item[0], size_left, item[2]])
                    size_left = 0
                    break
        order_book.asks = order_book.asks[sweep_ind:]
        ind = 0
        while price < order_book.bids[ind][0]:
            ind += 1
        if price == order_book.bids[ind][0]:
            order_book.bids[ind][1] += size_left
        else:
            order_book.bids = order_book.bids[:ind] + [[price, size_left]] + order_book.bids[ind:]
        return order_book

    def sell(self, order_book, price, size):
        size_left = size
        sweep_ind = 0
        for i, item in enumerate(order_book.bids):
            if price <= item[0] and item[2] not in self.open_orders.keys():
                if size_left >= item[1]:
                    sweep_ind = i+1
                    size_left -= item[1]
                    order_book.trades.append([item[0], item[1]])
                else:
                    order_book.bids[i][1] -= size_left
                    order_book.trades.append([item[0], size_left])
                    size_left = 0
                    break
        order_book.bids = order_book.bids[sweep_ind:]
        ind = 0
        while price > order_book.asks[ind][0]:
            ind += 1
        if price == order_book.asks[ind][0]:
            order_book.asks[ind][1] += size_left
        else:
            order_book.asks = order_book.asks[:ind] + [[price, size_left]] + order_book.asks[ind:]
        return order_book
'''