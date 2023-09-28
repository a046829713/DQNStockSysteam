class Backtest():
    def __init__(self, markerpostions: list, Symbol_data ,bars_count) -> None:
        """_summary_

        Args:
            # 由機器人所取得的部位
            markerpostions (list): [0, 0, 0, 0, 0, 0, 0....1,1,1,0,........]
            
            
            print(len(marketpostions)) # 2551
        """
        self.marketpostions = markerpostions
        self.Close = Symbol_data['close'].to_numpy()
        
        # 前面10個當樣本
        self.Close = self.Close[bars_count:]
        
        # 最後一個不計算
        self.Close = self.Close[:-1]
        
        # self.Close = self.Close[len(self.Close) - len(self.marketpostions):]
        self.ClosedPostionprofit = 500000
        self.sheets = 1
        self.marketpostion = 0
        self.lastmarketpostion = 0
        self.entryprice = 0
        self.exitsprice = 0
        self.slippage: float = 0.0025
        print(self.Count())

    def Count(self):
        for index, postion in enumerate(self.marketpostions):
            self.profit = 0.0
            self.Buy_Fee = 0.0
            self.Sell_Fee = 0.0
            self.tax = 0.0

            if index == 0:
                self.lastmarketpostion = 0
            else:
                self.lastmarketpostion = self.marketpostions[index - 1]

            if postion == 1 and self.lastmarketpostion == 0:
                self.entryprice = self.Close[index]
                # 添加滑價
                self.entryprice = self.entryprice * (1+self.slippage)

                self.Buy_Fee = self.entryprice * self.sheets * 1000 * 0.001425

            if postion == 0 and self.lastmarketpostion == 1:
                self.exitsprice = self.Close[index]
                self.exitsprice = self.exitsprice * (1-self.slippage)

                self.profit = (self.exitsprice -
                               self.entryprice) * 1000 * self.sheets

                self.Sell_Fee = self.exitsprice * self.sheets * 1000 * 0.001425
                self.tax = self.exitsprice * self.sheets * 1000 * 0.003

                self.entryprice = 0

            # if self.profit != 0 or self.Buy_Fee != 0:
            #     print("目前索引:", index, "目前部位:", postion, "目前損益:",
            #           self.profit, "買入手續費:", self.Buy_Fee,"賣出手續費:",self.Sell_Fee,"賣出稅:",self.tax)

            self.changeInTimeMoney(
                self.ClosedPostionprofit, self.profit, self.Buy_Fee, self.Sell_Fee, self.tax)

        return self.ClosedPostionprofit

    def changeInTimeMoney(self, cash: float, Profit: float, Buy_Fee: float, Sell_Fee: float, tax: float):
        """
            用來更新已平倉損益
        """
        self.ClosedPostionprofit = cash + Profit - Buy_Fee - Sell_Fee - tax