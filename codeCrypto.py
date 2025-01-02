import streamlit as st  
import pandas as pd  # I'll use it to recap the different options that are inputs 
import numpy as np  # vectorisation for quicker computation
import matplotlib.pyplot as plt   
import math as m  

# scipy was giving me problems to deploy the app, so I used the following for an equivalent estimate of norm.cdf()
def norm_cdf_math(x):
    return 0.5 * (1 + m.erf(x / m.sqrt(2)))

st.set_page_config(layout="wide")

st.title('Crypto Options Trading using Vanilla Options')

with st.expander('About this app'):
    st.write('This app enables you to plot the P&L at expiry of the combination of call and put options with different strikes. The (European) options premium is computed using the Black-Scholes pricing model. You can input the parameters in the left sidebar.')
with st.expander('Arbitrage Opportunities'):
    st.write('This app also enables users to find arbitrage opportunities in the market with the Black Scholes theoretical value call and put premiums')
#---------- Side bar -----------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        Created by <span style='color: #39FF14;'> Luc Villermain</span>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.header('Input Parameters')


asset = st.sidebar.selectbox("Select Asset", ["BTC", "ETH", "SOL"])


if asset == "BTC":
    spot_range = np.arange(60000, 150001, 5)
elif asset == "ETH":
    spot_range = np.arange(1000, 10001, 1)
elif asset == "SOL":
    spot_range = np.arange(50, 501, 1)

#-------- Input parameters---------
S = st.sidebar.selectbox('Spot price (S)', spot_range)
r = st.sidebar.selectbox('Risk-free Interest Rate %(r)', [round(x * 0.25, 2) for x in range(0, 41)])
T = st.sidebar.selectbox('Time to expiry in days (T)', range(1, 366)) / 365  
sigma = st.sidebar.selectbox('Volatility (%)', [round(x * 0.25, 2) for x in range(4, 321)])

#----------The inputs for r and sigma are percentages, /100 converts to decimals for the calculations------

sigma = sigma / 100
r = r / 100

#-----------spot and strike inputs based on asset-----------

# Ensure liste_options matches the size of spot_range
if 'liste_options' not in st.session_state or len(st.session_state.liste_options) != len(spot_range):
    st.session_state.liste_options = np.zeros(len(spot_range))

if 'data' not in st.session_state:
    st.session_state.data = []

#-----------Computing Call and Put premiums----------------
def price_call(S, K, sigma, T, r):
    d1 = (m.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * m.sqrt(T))
    d2 = d1 - sigma * m.sqrt(T)
    return S * norm_cdf_math(d1) - K * m.exp(-r * T) * norm_cdf_math(d2)

def price_put(S, K, sigma, T, r):
    return price_call(S, K, sigma, T, r) - S + K * m.exp(-r * T)  # Parité put-call

#-----------Computing P&L for calls and puts (lists of values)----------------
def put_function(K_put):
    liste_OTM = np.zeros(len(spot_range[spot_range >= K_put]))
    liste_ITM = np.array([K_put - i for i in spot_range[spot_range < K_put]])  
    PandL = np.concatenate((liste_ITM, liste_OTM)) - price_put(S, K_put, sigma, T, r)
    return PandL

def call_function(K_call):
    liste_OTM = np.zeros(len(spot_range[spot_range <= K_call]))
    liste_ITM = np.array([i - K_call for i in spot_range[spot_range > K_call]])
    PandL = np.concatenate((liste_OTM, liste_ITM)) - price_call(S, K_call, sigma, T, r)
    return PandL

# Pick a strike
selected_strike = st.selectbox('Pick the strike AND THEN click on the position you want with the latter ', spot_range)
st.write('(You can enter the strike using your keyboard) ')

# Display option prices
call_price = price_call(S, selected_strike, sigma, T, r)
put_price = price_put(S, selected_strike, sigma, T, r)
st.write(f"Call price: {call_price:.2f}")
st.write(f"Put price: {put_price:.2f}")

# 4 buttons 
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('Add Long Call'):
        st.session_state.liste_options += call_function(selected_strike)
        st.session_state.data.append(["Long Call", selected_strike])

with col2:
    if st.button('Add Short Call'):
        st.session_state.liste_options -= call_function(selected_strike)
        st.session_state.data.append(["Short Call", selected_strike])

with col3:
    if st.button('Add Long Put'):
        st.session_state.liste_options += put_function(selected_strike)
        st.session_state.data.append(["Long Put", selected_strike])

with col4:
    if st.button('Add Short Put'):
        st.session_state.liste_options -= put_function(selected_strike)
        st.session_state.data.append(["Short Put", selected_strike])

# RESET button
if st.button('RESET'):
    st.session_state.liste_options = np.zeros(len(spot_range))  # Reset to zero
    st.session_state.data = []  # Clear option data

# Recap table
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data, columns=["Option", "Strike"])
    st.write("Table Recap of positions:")
    st.dataframe(df)

st.header("P&L Graph of Combined Options")

X = spot_range
Y = st.session_state.liste_options

# Plotting
fig, ax = plt.subplots()
ax.plot(X, Y, color='black')

# Green and Red filling
ax.fill_between(X, Y, where=(Y > 0), color='green', alpha=0.6, interpolate=True)
ax.fill_between(X, Y, where=(Y < 0), color='red', alpha=0.6, interpolate=True)

# Horizontal line at y=0
ax.axhline(0, color='black', linewidth=1)

# Plot the final graph
st.pyplot(fig)
