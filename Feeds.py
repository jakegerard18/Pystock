import streamlit as st
import pandas as pd
import datetime
import requests
import time
from bs4 import BeautifulSoup


def app():
    # Add title
    st.write("""
    # Ticker Feed
    Select a stock to view live ticker data on that stock. 
    """)


    with open('./stock symbols.csv', 'r', encoding='utf-8-sig') as stock_file:
        stock_list = pd.read_csv(stock_file)
        symbols = stock_list.iloc[:, 0]
        selected = st.selectbox(label="", options=symbols)

    def scrape(symbol):
        #Get URL from symbol
        url = ('https://finance.yahoo.com/quote/') + symbol + ('?p=') + symbol + ('&.tsrc=fin-srch')
        r = requests.get(url)

        #Get the current price of the stock
        price_content = BeautifulSoup(r.text, 'lxml')
        price_content = price_content.find('div')
        price = price_content.find('span', {'class': 'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text

        #Get the change value of the stock price
        change_content = BeautifulSoup(r.text, 'lxml')
        change_content = change_content.find('div')
        change_content = change_content.find('span', {'class': 'Trsdu(0.3s) Fw(500) Pstart(10px) Fz(24px) C($negativeColor)'})
        if change_content != None:
            change = change_content.text
        else:
            change_content = BeautifulSoup(r.text, 'lxml')
            change_content = change_content.find('div')
            change_content = change_content.find('span', {'class': 'Trsdu(0.3s) Fw(500) Pstart(10px) Fz(24px) C($positiveColor)'})
            change = change_content.text

        #Get first table
        soup = BeautifulSoup(r.text, 'lxml')
        soup = soup.find('table', {'class':'W(100%)'})
        pc = soup.find('tbody').contents[0].td.next_sibling.text
        o = soup.find('tbody').contents[1].td.next_sibling.text
        b = soup.find('tbody').contents[2].td.next_sibling.text
        a = soup.find('tbody').contents[3].td.next_sibling.text
        dr = soup.find('tbody').contents[4].td.next_sibling.text
        ftwr = soup.find('tbody').contents[5].td.next_sibling.text
        v = soup.find('tbody').contents[6].td.next_sibling.text
        av = soup.find('tbody').contents[7].td.next_sibling.text

        #Get second table
        soup = BeautifulSoup(r.text, 'lxml')
        soup = soup.find('table', {'class':'W(100%) M(0) Bdcl(c)'})
        mc = soup.find('tbody').contents[0].td.next_sibling.text
        beta = soup.find('tbody').contents[1].td.next_sibling.text
        pe = soup.find('tbody').contents[2].td.next_sibling.text
        eps = soup.find('tbody').contents[3].td.next_sibling.text
        ed = soup.find('tbody').contents[4].td.next_sibling.text
        fdy = soup.find('tbody').contents[5].td.next_sibling.text
        edd = soup.find('tbody').contents[6].td.next_sibling.text
        t = soup.find('tbody').contents[7].td.next_sibling.text

        return price, change, pc, o, b, a, dr, ftwr, v, av, mc, beta, pe, eps, ed, fdy, edd, t

    if st.button('Get Ticker'):
        slot1 = st.empty()
        slot2 = st.empty()
        slot3 = st.empty()
        slot4 = st.empty()
        slot5 = st.empty()
        slot6 = st.empty()
        slot7 = st.empty()
        slot8 = st.empty()
        slot9 = st.empty()
        slot10 = st.empty()
        slot11 = st.empty()
        slot12 = st.empty()
        while True:
            price, change, pc, o, b, a, dr, ftwr, v, av, mc, beta, pe, eps, ed, fdy, edd, t = scrape(selected)
            slot1.header(selected + " : " + price + " " + change)
            slot2.text("Previous Close :" + pc + "\t\t" + "Open: " + o)
            slot3.text("Bid: " + b + "\t\t" + "Ask: " + a)
            slot4.text("Day Range: " + dr + "\t" + "52 Week Range: " + ftwr)
            slot5.text("Volume: " + v + "\t\t" + "Average Volume: " + av)
            slot6.text("Market Cap: " + mc + "\t\t" + "Beta: " + beta)
            slot7.text("PE Ration(TTM): " + pe + "\t\t" + "EPS(TTM): " + eps + "\n")
            slot8.text("")
            slot9.text("Earnings Date: " + ed)
            slot10.text("Forward Dividend & Yield: " + fdy)
            slot11.text("Ex-Dividend Date: " + edd)
            slot12.text("1 Year Target Est: " + t)


























