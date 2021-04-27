import numpy as np 
import os
import matplotlib.pyplot as plt

from numpy.core.fromnumeric import take 
from utils import *

n_marketmaker = 10
n_hft = 10
n_randomretailer = 100

def simulate(N=100, lf_ratio = 0.2):
    ob = Orderbook(100, 1)
    market_makers, hfts, retailers = [], [], []
    for i in range(n_marketmaker):
        market_makers.append(MarketMaker(100,1,1,10000))
    for i in range(n_hft):
        hfts.append(HFT(50,10000))
    for i in range(n_randomretailer):
        retailers.append(RandomRetailer(20, 0.3))
    run_sequences = []
    traders = [market_makers, hfts, retailers]
    for i in range(N):
        if np.random.random() < lf_ratio:
            run_sequences.append((2, int(np.random.random()*n_randomretailer)))
        else:
            if np.random.random() > 0.7:
                run_sequences.append((1, int(np.random.random()*n_hft)))
            else:
                run_sequences.append((0, int(np.random.random()*n_marketmaker)))
    prices = []
    for trader_class, trader_num in run_sequences:
        print(trader_class, trader_num)
        traders[trader_class][trader_num].run(ob)
        #ob.show(only_trade = False)
        try:
            if len(ob.trades)>0:
                prices.append(ob.trades[-1][0])
            else:
                prices.append(100)
            ob.quote()
        except:
            ob.show()
            print('Market Fucked up, exiting')
            break
    plt.plot(prices)
    plt.savefig('price')
    plt.close()
    ob.show()

simulate()