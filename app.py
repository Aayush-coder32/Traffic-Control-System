import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Smart City Traffic Analytics System",
                    page_icon="ðŸš¦",layout="wide")
st.title("ðŸš¸ Smart City Traffic Analytics System")
st.write("ðŸš” Analyze trafic data and predict traffic valume using mchine learining")
st.divider()

@st.cache_data
def data_load():
    df = pd.read_csv("DataSet/traffic.csv")
    return df
df = data_load()

st.subheader("ðŸ“ˆ Traffic Dataset")
if st.button("ðŸ‘ï¸click me for see Raw Data:",key="btn8"):
    st.dataframe(df)

st.subheader("ðŸ“ˆ DataSet Information")
if st.button("ðŸ‘ï¸click me for see :",key="btn9"):
    col1 , col2 = st.columns(2) 
    with col1:
        st.metric("**Total Row**",df.shape[0])
        st.metric("**Total Column**",df.shape[1])
    with col2:
        st.write("**Column Name**")
        st.write(df.columns.tolist())

st.subheader("ðŸ“Š Data Type")

datatype = pd.DataFrame(df.dtypes,columns=["Data Type"])
if st.button("ðŸ‘ï¸click me for see :",key="btn6"):
    st.dataframe(datatype)

st.subheader("ðŸ¤” Missing Values ")
if st.button("ðŸ‘ï¸click me for see :",key="btn5"):
    with st.container(border=True):
        missingvalue = pd.DataFrame(df.isnull().sum(),columns=["Missing Values"])
        st.dataframe(missingvalue)

st.subheader("ðŸ“ Statical Values")
if st.button("ðŸ‘ï¸click me for see :",key="btn7"):
    st.dataframe(df.describe())

st.subheader("ðŸ‘ï¸ Starting 10 Rows Value")
st.dataframe(df.head(10))

st.subheader("ðŸ‘€ Last 10 Rows Value")
st.dataframe(df.tail(10))

st.subheader("ðŸš¦Trafic Valume Distribution")
if st.button("ðŸš¦Trafic Valume Distribution :",key="btn1"):
    with st.container(border=True):
        fig,ax = plt.subplots(figsize=(5,2))
        ax.hist(df["traffic_volume"],bins=20,color="skyblue",edgecolor="black",linewidth=0.8)
        ax.set_title("Trafic Valume Distribution")
        ax.set_xlabel("Traffic valume")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

st.subheader("ðŸŒ¡ï¸ Temperature vs Trafic Valume")
if st.button("ðŸŒ¡ï¸ Temperature vs Trafic Valume :",key="btn2"):
    with st.container(border=True):
        fig,ax = plt.subplots(figsize=(4,2))
        ax.scatter(df["temp"],df["traffic_volume"],s=0.2,color="red")
        st.pyplot(fig)


st.subheader("ðŸ˜¶â€ðŸŒ«ï¸ Cloud vs Traffic")
if st.button("ðŸ˜¶â€ðŸŒ«ï¸ Cloud vs Traffic :",key="btn3"):
    with st.container(border=True):
        fig,ax = plt.subplots(figsize=(4,2))
        ax.scatter(df["clouds_all"],df["traffic_volume"],color="green",s=0.2)
        ax.set_xlabel("Cloud Coverage")
        ax.set_ylabel("Vehicals")
        st.pyplot(fig)

st.subheader("ðŸ”¥ Correlation Heatmap")
if st.button("ðŸ”¥ Correlation Heatmap :",key="btn4"):
    with st.container(border=True):
        numeric = df.select_dtypes(include=["number"])
        fig,ax = plt.subplots(figsize=(5,3))
        sns.heatmap(numeric.corr(),annot=True,cmap="coolwarm",ax=ax)
        st.pyplot(fig)

st.subheader("Data Preprocessing")
df = df.dropna()

@st.cache_data
def preprocess(df):
    encoder = LabelEncoder()
    df["weather_main"]=encoder.fit_transform(df["weather_main"])
    df["weather_description"]=encoder.fit_transform(df["weather_description"])
    df["date_time"]=pd.to_datetime(df["date_time"],dayfirst=True)
    df["Hours"]=df["date_time"].dt.hour
    df["Day"]=df["date_time"].dt.day
    df["Month"]=df["date_time"].dt.month
    df.drop("date_time",axis=1,inplace=True)
    return df
df = preprocess(df)

st.success("Preproccessing Complete")
st.subheader("Proccessed Data")
st.dataframe(df.head(5))

x = df.drop("traffic_volume",axis=1)
y = df["traffic_volume"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

@st.cache_data
def train_model(x_train,y_train):
    model =RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(x_train,y_train)
    st.success("Model Trained Successfully")
    return model
model = train_model(x_train,y_train)


pred = model.predict(x_test)
score = r2_score(y_test,pred)
st.metric("R2 Score :",score)

st.subheader("Smart City Traffic ui")
with st.container(border=True):
    temperature_c = st.number_input("ðŸŒ¡ï¸Temperature (C) :",min_value=-20.0,max_value=50.0,value=20.0,step=1.0)
    temp = temperature_c+ 272.15

    rain = st.slider("ðŸŒ¦ï¸ Rain (last in 1 hrs) :",0.0,20.0,0.0,0.1)
    snow =  st.slider("â„ï¸ Snow(last in 1 hrs) :",0.0,20.0,0.0,0.1)
    cloud = st.slider("â˜ï¸ Coverage (%) :",0,100,50)
    weather_main = st.selectbox("ðŸŒ«ï¸ Weather Main :",[0,1,2,3,4,5,6,7])
    weather_description = st.selectbox("â›… Weather_Description :",list(range(38)))
    hours = st.slider("â° Hours :",0,23,12)
    day = st.slider("â˜€ï¸ Days :",1,31,15)
    month = st.selectbox("ðŸ—“ï¸ Month :",list(range(1,13)))

    st.divider()
    if st.button("ðŸš• Predict Traffic Volume :",use_container_width=True):
        input_data =[[temp,rain,snow,cloud,weather_main,weather_description,hours,day,month]]

        prediction = model.predict(input_data)
        st.success(f"ðŸš” Prediction Traffic Valume :{prediction[0]:.0f}")
        st.info(f"ðŸŽ¯ Model Accurecy(R2 Score) :{score:.4f}")


