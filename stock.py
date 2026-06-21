# import streamlit as st 
# import pickle
# import numpy as np

# # load the file
# with open("stock_price_model.pkl","rb")as file : model = pickle.load(file)

# # ADD TITLE
# st.title("📈 Stock Price Prediction App")
# st.write("Predict stock closing prices using machine learning")  # use normally write something in tha webpage

# # add input boxes

# open_price = st.number_input("Open Price")
# high_price = st.number_input("High Price")
# low_price = st.number_input("Low Price")
# volume = st.number_input("Volume")


# # add remainig input boxes

# ma_50 = st.number_input("50-Day Moving Average")
# ma_100 = st.number_input("100-Day Moving Average")
# daily_return = st.number_input("Daily Return")
# volatility = st.number_input("Volatility")

# # add the predict button
# if st.button("Predict Price"):
#     st.write("Prediction button clicked!")
    
# # connect the model with button
# if st.button("Predict Price"):

#     features = np.array([[open_price,
#                           high_price,
#                           low_price,
#                           volume,
#                           ma_50,
#                           ma_100,
#                           daily_return,
#                           volatility]])

#     prediction = model.predict(features)

#     st.success(f"Predicted Close Price = ₹{prediction[0]:.2f}")
    
# # use sidebar for the input
# open_price = st.sidebar.number_input("Open Price")
# high_price = st.sidebar.number_input("High Price")
# low_price = st.sidebar.number_input("Low Price")
# volume = st.sidebar.number_input("Volume")
# ma_50 = st.sidebar.number_input("50-Day Moving Average")
# ma_100 = st.sidebar.number_input("100-Day Moving Average")
# daily_return = st.sidebar.number_input("Daily Return")
# volatility = st.sidebar.number_input("Volatility")

# # show prediction in a beautiful matric card
# st.metric(
#     label="Predicted Close Price",
#     value=f"₹{prediction[0]:.2f}"
# )
        
import streamlit as st
import pickle
import numpy as np
import yfinance as yf

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="IRCTC Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- Load Model ----------------
with open("stock_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- Title ----------------
st.title("📈 IRCTC Stock Price Prediction")
st.write("Predict closing price using Machine Learning")

# ---------------- Sidebar ----------------
st.sidebar.header("📊 Enter Stock Details")

open_price = st.sidebar.number_input(
    "Open Price",
    min_value=0.0,
    value=523.90,
    step=0.01
)

high_price = st.sidebar.number_input(
    "High Price",
    min_value=0.0,
    value=524.00,
    step=0.01
)

low_price = st.sidebar.number_input(
    "Low Price",
    min_value=0.0,
    value=518.55,
    step=0.01
)

volume = st.sidebar.number_input(
    "Volume",
    min_value=0,
    value=754494
)

# ---------------- Prediction ----------------
if st.button("🚀 Predict Price"):

    try:
        # Download IRCTC data
        data = yf.download("IRCTC.NS", period="5y")

        # Fix MultiIndex columns
        if isinstance(data.columns, type(data.columns)):
            try:
                data.columns = data.columns.get_level_values(0)
            except:
                pass

        # Feature Engineering
        data["MA_50"] = data["Close"].rolling(50).mean()
        data["MA_100"] = data["Close"].rolling(100).mean()
        data["Daily_Return"] = data["Close"].pct_change()
        data["Volatility"] = data["Daily_Return"].rolling(30).std()

        data.dropna(inplace=True)

        # Check empty dataframe
        if len(data) == 0:
            st.error("No stock data available.")
            st.stop()

        latest_row = data.iloc[-1]

        ma_50 = latest_row["MA_50"]
        ma_100 = latest_row["MA_100"]
        daily_return = latest_row["Daily_Return"]
        volatility = latest_row["Volatility"]

        # Features
        features = np.array([[
            open_price,
            high_price,
            low_price,
            volume,
            ma_50,
            ma_100,
            daily_return,
            volatility
        ]])

        # Prediction
        prediction = model.predict(features)

        st.success("Prediction Completed Successfully ✅")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="📈 Predicted Close Price",
                value=f"₹{prediction[0]:.2f}"
            )

        with col2:
            st.metric("MA_50", f"{ma_50:.2f}")
            st.metric("MA_100", f"{ma_100:.2f}")
            st.metric("Daily Return", f"{daily_return:.4f}")
            st.metric("Volatility", f"{volatility:.4f}")

    except Exception as e:
        st.error(f"Error : {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Developed by Gautam 🚀")
