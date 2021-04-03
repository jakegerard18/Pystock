#A note on sources: All code used to develop Pystock is either completely my own or was derived under the guidance
# of an online tutorial such as those offered by Krish Naik on Youtube(channel: Krish Naik) and Dr. Jason Brownlee
# of machinelearningmastery.com.

#Libraries
import Home
import Feeds
import Statistics
import Correlation
import Prediction
import streamlit as st


#Add navbar
PAGES = {
    "Home": Home,
    "Ticker Feed": Feeds,
    "Statistics": Statistics,
    "Prediction": Prediction,
    "Correlation": Correlation
}
st.beta_set_page_config(page_title='Pystock')
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)