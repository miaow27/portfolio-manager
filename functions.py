import numpy as np
import pandas as pd
import plotly.express as px
from functools import reduce
from io import StringIO, BytesIO
import base64
from datetime import date
import plotly.graph_objects as go
import requests
import json
import re

def get_api_stock_price_previous_close(s, type = 'stock'):
  
    key = "qiwvnAkCH9kez5KSxaticnXPcwoczKjh"

    with requests.Session() as ses:
        s = s.replace('-', '').strip().upper()

        if type == 'crypto':
            if re.search('USD', s):
                s = 'X:' + s
            else:
                s = 'X:' + s + 'USD'
        print(s)
        url = 'https://api.polygon.io/v2/aggs/ticker/' + s + '/prev?unadjusted=true&apiKey=' + key
        p = 0
        try:
            p = float(ses.get(url).json()['results'][0]['c']) # closing price
        except:
            pass
    return p

class Portfolio:
  
    def __init__(self, raw_portfolio):
        self.raw_portfolio = raw_portfolio

    def cln(self):
        # clean up raw files
        df = self.raw_portfolio.copy()
        df.columns = ['crypto', 'stock', 'date', 'buy-sell', 'price', 'share']
        df['date'] = pd.to_datetime(df['date'])
        df[['crypto', 'buy-sell', 'price', 'share']] = df[['crypto', 'buy-sell', 'price', 'share']].astype(float)
        df['share-sign'] = df['share'] * df['buy-sell'] * (-1)
        df['earning'] = df['share'] * df['price'] * df['buy-sell']
        df['stock_type'] = df['crypto'].apply(lambda x: 'crypto' if x == 1 else 'stock')
        self.portfolio = df
        
    def get_stock_price(self):
        # try to get price from API
        stock_price = {}
        s_list = self.portfolio[['stock', 'stock_type']].drop_duplicates()
        
        for index, row in s_list.iterrows():
            p = get_api_stock_price_previous_close(row['stock'], row['stock_type'])
            stock_price.update({row['stock']:p})
        
        self.stock_price = stock_price

    def summarize(self):
        # get total bought/seld shared, avg price
        # total earning = what goes into the bank - what goes out of the bank (regarless of price current value)
        
        def quick_summary(df):
            s = df.groupby('stock').agg(
                share=pd.NamedAgg('share-sign', 'sum'),
                value=pd.NamedAgg('earning', 'sum'),
                date_min=pd.NamedAgg('date', 'min'),
                date_max=pd.NamedAgg('date', 'max')
            ).reset_index()
            s['avg_price'] = s['value'] / s['share']
            s['days'] = (s['date_max'] - s['date_min'])/np.timedelta64(1,'D')
            s['date_min'] = s['date_min'].dt.date
            s['date_max'] = s['date_max'].dt.date
            return s[['stock', 'share', 'avg_price', 'value', 'date_min', 'date_max', 'days']]
        
        buy = quick_summary(self.portfolio[self.portfolio['buy-sell'] < 0])
        sel = quick_summary(self.portfolio[self.portfolio['buy-sell'] > 0])
        tot = quick_summary(self.portfolio)
        
        buy.iloc[:,1:4] = buy.iloc[:,1:4].abs() # share/earning to positive
        sel.iloc[:,1:4] = sel.iloc[:,1:4].abs()
        tot = tot.drop('avg_price', axis = 1)
        
        buy.columns = ['stock', 'buy_shares', 'buy_avg_price', 'buy_value', 'buy_date_min', 'buy_date_max', 'buy_days']
        sel.columns = ['stock', 'sell_shares', 'sell_avg_price', 'sell_value', 'sell_date_min', 'sell_date_max', 'sell_days']
        tot.columns = ['stock', 'remain_shares', 'current_earning', 'trade_day_min', 'trade_day_max', 'trade_days']
        
        self.buy = buy.round(2)
        self.sell = sel.round(2)
        self.tot = tot.round(2)
        
    def summarize_value(self, STOCK_PRICE_DICT):
        # total current earned value = if you sell 100% price to the stock price - what goes outside of bank
        
        val = pd.DataFrame(STOCK_PRICE_DICT.items()).rename(columns = {0:'stock',1:'today_price'})
        val = pd.merge(val, self.tot[['stock', 'remain_shares', 'current_earning']])
        val['remain_shares_value'] = val['remain_shares'] * val['today_price']
        val['total_earning'] = val['remain_shares_value'] + val['current_earning']
        
        return val.round(2)
    
    def stock_return(self, STOCK_PRICE_DICT):
        # return = total earing / total biu 
        
        rt = pd.DataFrame(STOCK_PRICE_DICT.items()).rename(columns = {0:'stock',1:'today_price'})
        rt = pd.merge(rt, self.tot[['stock', 'remain_shares', 'current_earning']])
        rt = pd.merge(rt, self.buy[['stock', 'buy_value']])
        rt['return ratio'] = rt['current_earning'] / rt['buy_value']
        rt = rt[['stock', 'remain_shares', 'buy_value', 'current_earning', 'return ratio']]

        return rt.round(3)
    
    
    def price_change(self, STOCK_PRICE_DICT):
        # compare pulged in price with (1) avg buy-in price (2) latest buy-in price
        
        p_change = pd.DataFrame(STOCK_PRICE_DICT.items()).rename(columns = {0:'stock',1:'today_price'})
        p_change = pd.merge(p_change, self.buy[['stock', 'buy_avg_price', 'buy_shares']], on = 'stock')  
        
        p_change['%change avg buy'] = (p_change['today_price'] - p_change['buy_avg_price'])/p_change['buy_avg_price']
        p_change['%change avg buy'] = p_change['%change avg buy'].apply(lambda x: str(round(x*100, 1)) + '%')
        
        p_change = p_change[['stock',  'today_price', 'buy_avg_price', '%change avg buy', 'buy_shares']]
        return p_change.round(2)
        

    def what_if(self,
                FOCUS_STOCK,
                MARKET_PRICE_MIN,
                MARKET_PRICE_MAX,
                MARKET_PRICE_POINTS):

        stock = self.portfolio[self.portfolio['stock'] == FOCUS_STOCK]
        info = self.tot[self.tot['stock'] == FOCUS_STOCK]

        N = info['remain_shares'].values[0]  # total shares
        Y = info['current_earning'].values[0]  # current earning

        if N <= 0:
            return None, 'No share left'

        result = []
        p_range = np.arange(MARKET_PRICE_MIN,
                            MARKET_PRICE_MAX,
                            (MARKET_PRICE_MAX-MARKET_PRICE_MIN)/MARKET_PRICE_POINTS)  # price range

        for n_prop in [0.25, 0.5, 0.75, 1]:
            result.append({
                '% remain-shares': [n_prop] * len(p_range),
                'shares': [n_prop * N] * len(p_range),
                'price': list(p_range),
                'value':[p*N*n_prop for p in p_range],
                'current earning':[Y] * len(p_range),
                'total earning': [p*N*n_prop + Y for p in p_range]})

        result = reduce(lambda a, b: pd.DataFrame(a).append(pd.DataFrame(b)), result)

        fig = px.line(result, x='price', y='total earning', color='% remain-shares',title=FOCUS_STOCK + ' - Total Earning', template='seaborn')
        if result['total earning'].min()* result['total earning'].max() < 0:
            fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="grey")
        fig.update_layout(width = 700)
        return fig


def get_stock_sample():

    return pd.read_csv(StringIO(
        '''
        Crypto, Stock,Date,Buy(-1) / Sell (1),Price per Share,Share
        1, BTC,4/20/2021,-1,56219.22,0.08895382
        1, BTC,4/26/2021,-1,53574.24,0.01866335
        1, BTC,4/27/2021,1,54964.71,0.00909675
        1, BTC,5/4/2021,1,54691.62,0.0182818
        1, DOGE,4/19/2021,-1,0.380287,1315
        1, DOGE,4/25/2021,-1,0.260847,766
        1, DOGE,5/4/2021,1,0.528122,947
        1, DOGE,5/4/2021,1,0.530023,941
        0, COUR,4/1/2021,-1,45.5,21.97812
        0, COUR,4/1/2021,-1,51.93,20.130406
        0, COUR,4/12/2021,-1,50.88,19.65412
        0, COUR,4/16/2021,-1,46.08,30
        0, COUR,4/26/2021,1,48.72,20
        0, COUR,5/4/2021,1,45.5,20
        '''
    ))


def check_upload_file(df):

    cols = ['crypto', 'stock', 'date', 'buy-sell', 'price', 'share']
    message = ''

    if df.shape[1] != 6:
        message += 'Fail: your data should have exactly 6 columns (crypto, stock, date, buy-sell, price, share)\n'
        
    try:
        if (df.iloc[:, 0].abs().max() == 1) & (df.iloc[:, 0].abs().min() == 0):
            pass
        else:
            message += 'Fail: your first column should only have 0, 1 (0 for stock, 1 for crypto currency)' + '\n'
    except:
        message += 'Fail: your first column should only have 0, 1 (0 for stock, 1 for crypto currency)' + '\n'

    try:
        pd.to_datetime(df.iloc[:, 2])
    except:
        message += 'Fail: your third column should be a valid date format' + '\n'

    try:
        if df.iloc[:, 3].abs().max() == 1:
            pass
        else:
            message += 'Fail: your fourth column should only have 1, -1 (1 for buy, -1 for sell)' + '\n'
    except:
        message += 'Fail: your fourth column should only have 1, -1 (1 for buy, -1 for sell)' + '\n'

    try:
        df.iloc[:, 4].astype(float)
    except:
        message += 'Fail: your fifth column should be a numeric format for the stock price (per share)' + '\n'

    try:
        df.iloc[:, 5].astype(float)
    except:
        message += 'Fail: your sixth column should be a numeric format for number of shares' + '\n'

    return message


def to_excel(df_dict):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    for k, v in df_dict.items():
        v.to_excel(writer, sheet_name=k, index = False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def download_link_sample(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe dict
    out: href string
    """
    csv = df.to_csv(index = False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="portfolio-sample.csv" target="_blank">Download Template csv file</a>'

    return href


def download_link_summary(df_dict):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df_dict)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="portfolio-summary.xlsx">Download Portfolio Summary file</a>'


def beautiful_tbl(df, col_name=None, 
    bgcolor = "LightSteelBlue", hd_color = 'royalblue', col_width = 90, row_width = 30):
    
    if col_name == None:
        col_name = df.columns
    n = df.shape[1]
        
    fig = go.Figure(data=[go.Table(
    columnorder = list(range(1, n+1)),
    columnwidth = [col_width]*(n),
    header = dict(
        values = col_name,
        line_color='darkslategray',
        fill_color=hd_color,
        align=['left','center'],
        font=dict(color='white', size=15),
        height=row_width+15
    ),
    cells=dict(
        values=df.T.values,
        line_color='darkslategray',
        fill=dict(color=['paleturquoise', 'white']),
        align=['left', 'center'],
        font_size=15,
        height=row_width)
        )
    ])
    h = row_width *(df.shape[0]) + (row_width + 15) *1.5 + 20
    if (1.5 + n) * col_width < 1000:
        w = (1.5 + n) * col_width
    else:
        w = 1000
    fig.update_layout(height = h, width = w, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=bgcolor)
    
    return fig