#Libraries
import streamlit as st
import pandas as pd
import pandas_datareader as dr
from datetime import datetime
from datetime import timedelta



def app():
    # Add title and image
    st.write("""
    # Statistics
    Select a stock and date range to view statistics on that stock.
    """)

    #Create a sidebar header
    st.sidebar.header('Date Range:')

    #Create a function to get the users input
    def get_input():
        start_date = st.sidebar.date_input("Starting Date:", value=(datetime.today() - timedelta(days=365)), min_value=datetime(1817, 3, 8), max_value=datetime.today())
        end_date = st.sidebar.date_input("Ending Date:", min_value=datetime(1817, 3, 8), max_value=datetime.today())

        with open('./Logo and Stock Symbols/stock symbols.csv', 'r') as stock_file:
            stock_list = pd.read_csv(stock_file)
            symbols = stock_list.iloc[:, 0]
            selected = st.selectbox(label="", options=symbols)
            index = stock_list[stock_list['ï»¿Symbol'] == selected].index.values
            stock_symbol = stock_list['ï»¿Symbol'][index].to_string(index=False)
            company_name = stock_list['Name'][index].to_string(index=False)
            sector = stock_list['Sector'][index].to_string(index=False)

        return start_date, end_date, stock_symbol.strip(), company_name, sector

    #Get stock data within time frame entered by the user
    def get_data(stock_symbol, start_date, end_date):

        #Get the date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        #Load the data
        df = dr.DataReader(stock_symbol, data_source='yahoo', start=start, end=end)
        df.reset_index()

        #Set the start and end index rows both to 0
        start_row = 0
        end_row = 0

        for i in range(0, len(df)):
            if start <= pd.to_datetime(df.index[i]):
                start_row = i
                break
        for j in range(0, len(df)):
            if end >= pd.to_datetime(df.index[i]):
                end_row = len(df) - 1 - j
                break

        df = df.set_index(pd.DatetimeIndex(df.index.values))

        return df.iloc[start_row:end_row + 1, :]



    #Get user input
    start_date, end_date, stock_symbol, company_name, sector = get_input()


    #Get the data
    df = get_data(stock_symbol, start_date, end_date)


    #Display stock name and sector header
    st.header("Company:" + company_name + "\n")
    st.header("Market Sector:" + sector + "\n")

    #Display the close price
    st.header("Closing Prices from " + str(start_date) + " to " + str(end_date))
    st.line_chart(df['Close'])

    #Display the volume
    st.header("Volume from " + str(start_date) + " to " + str(end_date))
    st.line_chart(df['Volume'])

    #Get statistics on the data
    st.header("Statistics from " + str(start_date) + " to " + str(end_date))
    st.write(df.describe())

