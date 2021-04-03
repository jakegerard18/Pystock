import streamlit as st
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import pandas_datareader as dr
from datetime import datetime
from datetime import timedelta
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import callbacks


def app():

    st.write("""
    # Prediction
    Enter a stock, date range, and training parameters to train a prediction model to generate prediction curves on that stock.
    """)

    #Create a sidebar header
    st.sidebar.header('Training Parameters:')

    def get_input():
        with open('./Logo and Stock Symbols/stock symbols.csv', 'r') as stock_file:
            stock_list = pd.read_csv(stock_file)
            symbols = stock_list.iloc[:, 0]
            selected = st.selectbox(label="", options=symbols)
            index = stock_list[stock_list['ï»¿Symbol'] == selected].index.values
            stock_symbol = stock_list['ï»¿Symbol'][index].to_string(index=False)
            company_name = stock_list['Name'][index].to_string(index=False)
        start_date = st.sidebar.date_input("Starting Date:", value=(datetime.today() - timedelta(days=365*3)), min_value=datetime(1817, 3, 8), max_value=datetime.today())
        end_date = st.sidebar.date_input("Ending Date:", min_value=datetime(1817, 3, 8), max_value=datetime.today())
        epochs = st.sidebar.text_input("Enter Number of Epochs:", 100)
        lag = st.sidebar.text_input("Enter Time Series Lag:", 30)
        steps = st.sidebar.text_input("Enter Training Steps:", 100)
        days = st.sidebar.text_input("Enter Prediction Length in Days:", 30)
        pullback = st.sidebar.text_input("Pullback (Used for experiments):", 0)
        return start_date, end_date, stock_symbol.strip(), company_name, int(epochs), int(lag), int(steps), int(days), int(pullback)

    global RMSE
    start_date, end_date, stock_symbol, company_name, epochs, lag, steps, days, pullback = get_input()

    ###Load and shape data
    df = dr.DataReader(stock_symbol, data_source='yahoo', start=start_date, end=end_date)
    prices = df.reset_index()['Adj Close']
    scaler = MinMaxScaler(feature_range=(0,1))
    prices = scaler.fit_transform(np.array(prices).reshape(-1,1))

    ###Split data into training and testing sets
    training_size = int(len(prices)*.65)
    training_data = prices[0:training_size, :]
    test_data = prices[training_size:len(prices), :]


    ###function to create time series datasets
    def create_timeseries(data, lag):
        t_series = TimeseriesGenerator(data, data, length=lag, batch_size=len(data))
        for i in range(len(t_series)):
            x, y = t_series[i]
        return np.array(x), np.array(y)

    ###generate time series for training and test sets
    x_train, y_train = create_timeseries(training_data, lag)
    x_test, y_test = create_timeseries(test_data, lag)

    ###reshape data for LSTM which requires 3D data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    ###create stacked LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    if st.button("Train Model"):
        bar = st.progress(0)
        class Callback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                bar.progress(((epoch+1)/epochs))

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64, verbose=1, callbacks=[Callback()])

        ###run trained model on train and test data, get RMSE, undo scaling
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)
        math.sqrt(mean_squared_error(y_train, train_predict))
        RMSE = math.sqrt(mean_squared_error(y_test, test_predict))
        train_predict = scaler.inverse_transform((train_predict))
        test_predict = scaler.inverse_transform((test_predict))


        ###Reformat data to be plotted
        prices = scaler.inverse_transform(prices)
        prices = np.array(prices).reshape(len(prices))
        train_predict = np.array(train_predict).reshape(len(train_predict))
        test_predict = np.array(test_predict).reshape(len(test_predict))


        ###Load data into single dataframe and plot
        global  data_df
        data_df = pd.DataFrame(columns=['Prices', 'Training Predictions', 'Testing Predictions'])
        data_df['Prices'] = prices
        data_df['Training Predictions'][(lag - 5):len(train_predict) + (lag - 5)] = train_predict
        data_df['Testing Predictions'][len(train_predict) + ((lag*2)-5):len(test_predict) + len(train_predict) + ((lag*2)-5)] = test_predict

        st.header(company_name + " Training Results from " + str(start_date) + " to " + str(end_date) + "\n")
        st.line_chart(data_df)
        st.text("Epochs used: " + str(epochs))
        st.text("Steps: " + str(steps))
        st.text("Training RMSE: " + str(RMSE))

    if st.button("Run Predictions"):
        ###Use all data to train for predictions
        data = prices[:]


        ###function to create time series datasets
        def create_timeseries(data, lag):
            t_series = TimeseriesGenerator(data, data, length=lag, batch_size=len(data))
            for i in range(len(t_series)):
                x, y = t_series[i]
            return np.array(x), np.array(y)

        ###generate time series for training and test sets
        x_train, y_train = create_timeseries(data, lag)

        ###reshape data for LSTM which requires 3D data
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        ###create stacked LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        bar = st.progress(0)

        class Callback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                bar.progress(((epoch + 1) / epochs))

        model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1,
                      callbacks=[Callback()])


        ###Forecasting function takes number of steps (or lag days) to use to make prediction, and number of days to forecast out.
        def forecast(data, steps, days):
            n_steps = steps
            x_input = data[-steps:].reshape(1,-1)
            temp_input = list(x_input)
            temp_input=temp_input[0].tolist()
            lst_output = []
            i = 0
            while (i < days):

                if (len(temp_input) > n_steps):
                    x_input = np.array(temp_input[1:])
                    x_input = x_input.reshape(1, -1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    temp_input = temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i = i + 1
                else:
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i = i + 1

            return lst_output
        lst_output = forecast(data[0:len(data)-pullback], steps, days)
        if(pullback != 0):
            prediction_RMSE = math.sqrt(mean_squared_error(data[len(data)-pullback:], lst_output))

        ###Convert data to dataframes and chart zoomed in view
        prices_df = pd.DataFrame(columns= ['Actual', 'Predictions'])
        prices_df['Actual'] = data_df['Prices']
        lst_output = scaler.inverse_transform(lst_output)
        lst_output = np.array(lst_output).reshape(len(lst_output))
        lst_output_df = pd.DataFrame(columns=['Actual','Predictions'])
        lst_output_df['Predictions'] = lst_output
        chart_df = pd.DataFrame(columns=['Actual', 'Predictions'])
        if(pullback == 0):
            chart_df = pd.concat([prices_df[['Actual', 'Predictions']], lst_output_df[['Actual', 'Predictions']]], ignore_index=True)
        else:
            chart_df['Actual'] = prices_df['Actual']
            chart_df['Predictions'] = np.nan
            chart_df['Predictions'][-pullback:] = lst_output_df['Predictions']

        st.header(company_name + " Prediction (Zoomed In)" + " from " + str(start_date) + " to " + str(end_date) + " for " + str(days) + " days\n")
        st.line_chart(chart_df[-(steps+days):])

        ###Chart zoomed out view
        st.header(company_name + " Prediction (Zoomed Out)" + " from " + str(start_date) + " to " + str(end_date) + " for " + str(days) + " days\n")
        st.line_chart(chart_df)
        st.text("Epochs: " + str(epochs))
        st.text("Steps: " + str(steps))
        if(pullback != 0):
            st.text("Prediction RMSE: " + str(prediction_RMSE))









