#Libraries
import streamlit as st
import numpy as np
import pandas as pd
import pandas_datareader as dr
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plot
import matplotlib
matplotlib.use('Agg')
import seaborn


def app():
    st.title('Stock Correlations')
    st.write('Select stocks to calculate their correlation:')
    global start_date
    global stock_list
    global selected

    st.sidebar.header("Starting Date:")
    start_date = st.sidebar.date_input("", value=(datetime.today() - timedelta(days=365 * 3)),
                                       min_value=datetime(1817, 3, 8), max_value=datetime.today())

    with open('./Logo and Stock Symbols/stock symbols.csv', 'r') as stock_file:
        stock_list = pd.read_csv(stock_file)
        symbols = stock_list.iloc[:, 0]
        selected = st.multiselect(label='', options=symbols)

    if st.button('Run Correlations'):
        # Array to store stock data
        data = []

        #Get closing prices for each stock
        for stock_symbol in selected:
            r = dr.DataReader(stock_symbol, data_source='yahoo', start=start_date)
            # add a symbol column
            r['Symbol'] = stock_symbol
            data.append(r)

        #Concatenate into df
        df = pd.concat(data)
        df = df.reset_index()
        df = df[['Date', 'Close', 'Symbol']]

        #Pivot data
        df_pivot = df.pivot('Date', 'Symbol', 'Close').reset_index()

        #Run correlations
        corr_df = df_pivot.corr(method='pearson')

        #Reset symbol as index
        corr_df.head().reset_index()

        #Generate plot
        seaborn.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0, linewidths=2.5)
        plot.yticks(rotation=0)
        plot.xticks(rotation=90)
        fig = plot
        st.header("Heatmap:")
        st.pyplot(fig)

        # Display data below chart
        st.header("Correlation Data:")
        st.write(corr_df)
        st.header("Closing Prices:")
        st.write(df_pivot)


