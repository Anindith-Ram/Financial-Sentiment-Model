import argparse
import os
import backtrader as bt
import pandas as pd
from datetime import datetime

class SentimentStrategy(bt.Strategy):
    params = (
        ('threshold', 0.5),
        ('hold_days', 5),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.composite_score = self.datas[0].composite_score
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.hold_days = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.composite_score[0] > self.params.threshold:
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                self.order = self.buy()
                self.hold_days = 0
        else:
            self.hold_days += 1
            if self.hold_days >= self.params.hold_days:
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                self.order = self.sell()

class SentimentData(bt.feeds.PandasData):
    lines = ('composite_score',)
    params = (('composite_score', -1),)

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest trading strategies using sentiment scores.')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy to backtest')
    parser.add_argument('--data_path', type=str, required=True, help='Path to sentiment data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save backtest results')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SentimentStrategy)
    df = pd.read_parquet(args.data_path)
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    data = SentimentData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcommission(commission=0.001)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()

if __name__ == '__main__':
    main() 