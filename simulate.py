import numpy as np 
import os
import matplotlib.pyplot as plt

import argparse
from numpy.__config__ import get_info

from numpy.testing._private.utils import runstring

parser = argparse.ArgumentParser(description='Trading simulation')
parser.add_argument('--slice-num', type=int, default=0,
                    help='training data number')
parser.add_argument('--lf-ratio', type=float, default=1,
                    help='low frequency investor ratio')
parser.add_argument('--mm-ratio', type=float, default=1,
                    help='momentum retailer')
parser.add_argument('--marketmaker-volume', type=float, default=6,
                    help='market maker volume')
parser.add_argument('--momentum-volume', type=float,default=6,
                    help='momentum volume')
parser.add_argument('--model-path', type=str, default='',
                    help='path to the hft model')
parser.add_argument('--thres', type=int, default=5,
                    help='threshold for spoofing')
parser.add_argument('--flat-time', type=int, default=300,
                    help='flat time for spoofers')
parser.add_argument('--fake-level', type=int, default=2,
                    help='at which level to put fake order')
parser.add_argument('--hft', dest='have_hft', action='store_true',
                    help='whether we have hft traders')
parser.add_argument('--spoofer', dest='have_spoofer', action='store_true',
                    help='whether we have spoofers')
parser.add_argument('--train', dest='train', action='store_true',
                    help='whether save trade history as training data')
args = parser.parse_args()

if args.model_path == '':
    args.model_path = f'params_{args.lf_ratio}_{args.mm_ratio}_{args.marketmaker_volume}_{args.momentum_volume}.npy'

from numpy.core.fromnumeric import diagonal, take
from numpy.core.getlimits import MachArLike
from numpy.lib.histograms import _hist_bin_fd 
from utils import *



n_marketmaker = 30
n_hft = 6
n_spoofer = 1
n_randomretailer = 20
n_maretailer = 5
n_mmretailer = 3

prs = np.array([8.34359179e-05,6.46337959e-05,4.46498508e-05,-9.01993149e-05,-5.83645289e-05,-4.34003853e-05 -6.98747574e-03])

def get_info(mms, thres=args.thres):
    momentum_left = 0
    for mm in mms:
        momentum_left += mm.left
    if momentum_left > thres:
        return 1
    if momentum_left < -1 * thres:
        return -1
    return 0


def simulate(N=50000, lf_ratio = 0.1, mm_ratio = 0.1, marketmaker_volume=50, momentum_volume=50, hft_ratio=0.5, model_path=''):
    tag = f'{args.slice_num}_{args.have_hft}_{args.have_spoofer}'
    ob = Orderbook(1000, 1, tag)
    market_makers, hfts, retailers, mm_retailers, spoofers = [], [], [], [], []
    print(f'Loading model from path {model_path}')
    if len(model_path) > 0:
        model = Model(np.load(model_path))
    else:
        model = None
    for i in range(n_marketmaker):
        market_makers.append(MarketMaker(marketmaker_volume,1,1,0,i))
    for i in range(n_hft):
        hfts.append(HFT(40, 0, i, model))
    for i in range(n_spoofer):
        spoofers.append(Spoofer(50, 100000,args.flat_time, args.fake_level, i))
    for i in range(n_randomretailer):
        retailers.append(RandomRetailer(100, 1, i))
    for i in range(n_maretailer):
        retailers.append(MARetailer(300, 8, 3, 0.05, i))
    for i in range(n_mmretailer):
        mm_retailers.append(MomentumRetailer(momentum_volume, 5, i))
    run_sequences = [(0,i%n_marketmaker) for i in range(5*n_marketmaker)]
    run_sequences += [(2,0)]
    run_sequences += [(2,i) for i in range(n_randomretailer, n_randomretailer+n_maretailer)]
    traders = [market_makers, hfts, retailers, mm_retailers, spoofers]
    print(f'HFT: {args.have_hft}, Spoofer: {args.have_spoofer}')
    for i in range(N):
        if np.random.random() < lf_ratio:
            run_sequences.append([2, int(np.random.random()*len(retailers))])
        else:
            if np.random.random() < mm_ratio:
                run_sequences.append([3, int(np.random.random()*n_mmretailer)])
            else:
                run_sequences.append([0, int(np.random.random()*n_marketmaker)])
                if np.random.random() < hft_ratio:
                    if args.have_spoofer:
                        run_sequences.append([4, int(np.random.random()*n_spoofer)])
                    if args.have_hft:
                        run_sequences.append([1, int(np.random.random()*n_hft)])
    run_sequences = np.array(run_sequences)
    #print(len(run_sequences))
    prices = []
    spoofs = []
    slices = []
    market_spreads = []
    for trader_class, trader_num in run_sequences:
        if trader_class == 4:
            info = get_info(mm_retailers)
            if info != 0 and not traders[trader_class][trader_num].have_info:
                traders[trader_class][trader_num].set_info(info, ob.time)
                is_spoof = info
        market_spreads.append(ob.spread())
        current_bid, current_ask = ob.quote()
        current_mid = (current_bid + current_ask) / 2.0
        #print(trader_class, trader_num, current_mid)
        ob = traders[trader_class][trader_num].run(ob)
        #ob.show(only_trade = False)
        try:
            if args.train:
                slices.append(ob.generate_slice())
            if len(ob.trades)>0:
                prices.append(ob.trades[-1][0])
            else:
                prices.append(1000)
            ob.quote()
        except:
            ob.save_history()
            #ob.show()
            print('Market Fucked up, exiting')
            break
    ob.save_history()
    prices = np.array(prices)
    #print('prices shape: ', prices.shape, spoofs.shape)
    market_spreads = np.array(market_spreads)
    plt.hist(market_spreads[market_spreads<6])
    plt.savefig(f'market_spread_{tag}')
    plt.close()
    if args.train:
        slices = np.array(slices)
        np.save(f'train_data_{args.slice_num}_{int(args.lf_ratio)}_{int(args.mm_ratio)}_{int(args.marketmaker_volume)}_{int(args.momentum_volume)}.npy', slices)
    print('Market maker status: ')
    for mm in market_makers:
        mm.show()
    plt.plot(prices, linewidth = 0.5)
    plt.title('last price chart')
    #print(np.array(range(len(prices)))[spoofs!=0].shape)
    #print(prices[spoofs!=0].shape)
    #plt.scatter(np.array(range(len(prices)))[spoofs!=0],prices[spoofs!=0], marker='o',c='r',s=0.1)
    plt.savefig('price_'+tag,dpi=1000)
    plt.close()
    #ob.show()
    sp_pnls, mm_pnls, hft_pnls, rr_pnls, mr_pnls = [], [], [], [], []
    sp_volumes, mm_volumes, hft_volumes, rr_volumes, mr_volumes = [], [], [], [], []
    poss = []
    print('Marketmaker pnl:')
    for mm in market_makers:
        mm_pnls.append(mm.update_pnl(ob))
        mm_volumes.append(mm.trading_volume)
        print(mm.position, mm.cash)
        poss.append(mm.position)
    print('HFT pnl:')
    for hft in hfts:
        hft_pnls.append(hft.update_pnl(ob))
        hft_volumes.append(hft.trading_volume)
        print(hft.position, hft.cash)
        poss.append(hft.position)
    print('Spoofer pnl:')
    for sp in spoofers:
        sp_pnls.append(sp.update_pnl(ob))
        sp_volumes.append(sp.trading_volume)
        print(sp.position, sp.cash)
        poss.append(sp.position)
    print('random retailer pnl:')
    for rr in retailers[:n_randomretailer]:
        rr_pnls.append(rr.update_pnl(ob))
        rr_volumes.append(rr.trading_volume)
        print(rr.position, rr.cash)
        poss.append(rr.position)
    print('ma retailer pnl:')
    for mr in retailers[n_randomretailer:]:
        mr_pnls.append(mr.update_pnl(ob))
        mr_volumes.append(mr.trading_volume)
        print(mr.position, mr.cash)
        poss.append(mr.position)
    print(np.mean(mm_pnls), mm_pnls)
    print(np.mean(hft_pnls), hft_pnls)
    print(np.mean(sp_pnls), sp_pnls)
    print(np.mean(rr_pnls), rr_pnls)
    print(np.mean(mr_pnls), mr_pnls)
    sp_volumes, mm_volumes, hft_volumes, rr_volumes, mr_volumes = np.array(sp_volumes), np.array(mm_volumes), np.array(hft_volumes), np.array(rr_volumes), np.array(mr_volumes)
    print(np.mean(mm_pnls/mm_volumes), mm_pnls/mm_volumes)
    if args.have_hft:
        print(np.mean(hft_pnls/hft_volumes), hft_pnls/hft_volumes)
    if args.have_spoofer:
        print(np.mean(sp_pnls/sp_volumes), sp_pnls/sp_volumes)
    print(np.mean(rr_pnls/rr_volumes), rr_pnls/rr_volumes)
    print(np.mean(mr_pnls/mr_volumes), mr_pnls/mr_volumes)
    print(np.sum(poss))

simulate(50000, args.lf_ratio/10, args.mm_ratio/10, args.marketmaker_volume*10, args.momentum_volume*10, 0.1, args.model_path) 