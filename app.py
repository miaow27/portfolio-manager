import streamlit as st 
import os
from io import StringIO
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from functions import check_upload_file, download_link_sample, download_link_summary, get_stock_sample, Portfolio, beautiful_tbl, get_stock_price_today

st.markdown('# Step 1: Upload your Stock Portfolio')
st.markdown('Note: it has to follow a certain format. Please download our template below')
st.markdown(download_link_sample(get_stock_sample()), unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your Stock Portfolio")

if uploaded_file is not None:
    
    bytes_data = uploaded_file.getvalue()
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    dataframe = pd.read_csv(uploaded_file)
    message = check_upload_file(dataframe)
    
    if len(message) > 0:
        st.error(message)
    else:
        st.warning('your data pass our test!')
        st.markdown('_Preview your Data_')
        st.dataframe(dataframe.head(3))
else:
    dataframe = get_stock_sample()
    st.warning('using our template data')
    st.dataframe(dataframe.head(3))


p = Portfolio(dataframe)
p.cln()
p.summarize()
p.latest_buy()
stock_lists = p.tot['stock'].unique()
stock_price = {}

stock_price_today, message = get_stock_price_today(stock_lists)
st.sidebar.markdown('### Stock Price Today')
st.sidebar.write('Stock successfuly get from Polygon API: ', message)
st.sidebar.write('Default is first tried to get from API then with maxium of your data')
for i in range(len(stock_lists)):
    s = stock_lists[i]
    default_val = stock_price_today[i]
    if default_val == 0:
        default_val = p.portfolio[p.portfolio['stock'] == s]['price'].round().max()
    v = st.sidebar.text_input(label = s, value = default_val)
    stock_price.update({s:float(v)})

st.markdown('# Step 2: Portfolio Summary')
st.markdown(download_link_summary({'overall':p.summarize_value(stock_price), 
                                   'price-change':p.price_change(stock_price),
                                   'overall-raw':p.tot,
                                   'buy':p.buy, 
                                   'sell':p.sell}), unsafe_allow_html=True)

st.markdown('## % Price Change')
st.plotly_chart(beautiful_tbl(p.price_change(stock_price), 
                col_name = ['Stock', 'Price Today', 'Latest Buy Date','Price Latest Buy', 'Share Latest Buy', 'Price Avg Buy', 'Share Total Buy', '% Price Latest Buy', '% Price Avg Buy'],
                col_width = 100, row_width = 40))

st.markdown('## Overall Summary')
st.plotly_chart(beautiful_tbl(p.summarize_value(stock_price), 
                col_name = ['Stock', 'Price Today', 'Remaining Shares', 'Current Earning', 'Remaining Share Values', 'Total Earning']))



with st.beta_expander('Compare Buy and Sell:'):
    if st.checkbox('Shares'):
        c1, c2 = st.beta_columns(2)
        tbl = p.buy[['stock', 'buy_shares']]
        tbl = tbl.merge(p.sell[['stock', 'sell_shares']])
        tbl['diff'] = (tbl['buy_shares'] - tbl['sell_shares']).round(2)
        c1.plotly_chart(beautiful_tbl(tbl, col_name = ['Stock', 'Buy Shares', 'Sell Shares', 'Remain Shares']))
    if st.checkbox('Total Values'):
        c1, c2 = st.beta_columns(2)
        tbl = p.buy[['stock', 'buy_value']]
        tbl = tbl.merge(p.sell[['stock', 'sell_value']])
        tbl['diff'] = (-tbl['buy_value'] + tbl['sell_value']).round(2)
        c1.plotly_chart(beautiful_tbl(tbl, col_name = ['Stock', 'Buy Total Values', 'Sell Total Values', 'Gained Values']))
    if st.checkbox('Avg Price'):
        c1, c2 = st.beta_columns(2)
        tbl = p.buy[['stock', 'buy_avg_price']]
        tbl = tbl.merge(p.sell[['stock', 'sell_avg_price']])
        tbl['diff'] = (-tbl['buy_avg_price'] + tbl['sell_avg_price']).round(2)
        c1.plotly_chart(beautiful_tbl(tbl, col_name = ['Stock', 'Buy Avg Price', 'Sell Avg Price', 'Difference in Avg Price']))

st.markdown('# Step 3: My Total Earning if I sell my stock?')
st.markdown('### _a what-if analysis to understand total earning base on range of market prices_')
st.write('Help anwer questions like this: ')
st.write('- How much is total earning if I sell 100%, 75%, 50%, 25% of my remaining shares?')
st.write('- How much shares do I have to sell to break even?')
st.write('- Minimum Market Price for me to earn $1000?')

c1, c2, c3 = st.beta_columns(3)
s = c1.selectbox('Pick a Stock', stock_lists, 0)
default_min = p.portfolio[p.portfolio['stock'] == s]['price'].round().min() * 0.5
p_min = float(c2.text_input('Min Price', default_min))
default_max = p.portfolio[p.portfolio['stock'] == s]['price'].round().max() * 2
p_max = float(c3.text_input('Max Price', default_max))

st.markdown(f'_You picked {s} and assume the market price will range between ${p_min} to ${p_max}_')
st.plotly_chart(p.what_if(s, p_min, p_max, 5))