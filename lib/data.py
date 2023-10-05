import pandas as pd
import collections
import numpy as np

Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close'])

class DataProvider():
    def __init__(self) -> None:
        self.data_path  = "data/TaiwanStockHistoryDailyData.csv"
        self.target_symobl = '2387'
        self.target_symobls = ['2330','2317','2409','3481','3032']
    
    def get_net_work_data(self) -> dict:
        """
            用來取得類神經網絡所需要的資料
            
        """
        out_dict = {}
        df = pd.read_csv(self.data_path)
        df = df[['stock_id','date','開盤價', '最高價', '最低價', '收盤價']]
        df = df.rename(columns={"開盤價":'open',  "最高價":"high", "最低價":"low" , "收盤價":"close"})
        
        for each_symbol in self.target_symobls:
            _df =  df[df['stock_id'] == each_symbol]
            _df =  _df[['date','open', 'high', 'low', 'close']]
            array_data = _df.values
            
            out_price = self.prices_to_relative(Prices(open=array_data[:,1],
                  high=array_data[:,2],
                  low=array_data[:,3],
                  close=array_data[:,4],
                  ))
            
            out_dict.update({each_symbol :out_price})
        
        return out_dict
    
    def get_real_data(self):
        df = pd.read_csv(self.data_path)
        df = df[df['stock_id'] == self.target_symobl]
        df = df[['date','開盤價', '最高價', '最低價', '收盤價']]
        df = df.rename(columns={"開盤價":'open',  "最高價":"high", "最低價":"low" , "收盤價":"close"})
        return df
        
    def load_relative(self):
        df = pd.read_csv(self.data_path)
        df = df[df['stock_id'] ==self.target_symobl]
        df = df[['date','開盤價', '最高價', '最低價', '收盤價']]
        df = df.rename(columns={"開盤價":'open',  "最高價":"high", "最低價":"low" , "收盤價":"close"})
        
        array_data = df.values

        return self.prices_to_relative(Prices(open=array_data[:,1],
                  high=array_data[:,2],
                  low=array_data[:,3],
                  close=array_data[:,4],
                  ))
        
    def prices_to_relative(self,prices):
        """
        Convert prices to relative in respect to open price
        :param ochl: tuple with open, close, high, low
        :return: tuple with open, rel_close, rel_high, rel_low
        """
        assert isinstance(prices, Prices)
        rh = (prices.high - prices.open) / prices.open
        rl = (prices.low - prices.open) / prices.open
        rc = (prices.close - prices.open) / prices.open
        return Prices(open=prices.open, high=rh, low=rl, close=rc)


if __name__ =="__main__":
    app = DataProvider()
    app.get_net_work_data()