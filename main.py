# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import KNNImputer
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import plotly.express as px
from prophet import Prophet

# Function to load data (5 pts)
def load_data(file_path, sheet_nam):
    # Load data from the CSV file or another format and return data
    data = pd.read_excel(file_path, index_col="date")
    return data


# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    data = data.drop('Facility', axis=1)
    knn = KNNImputer(n_neighbors=2)
    data['Value'] = knn.fit_transform(np.array(data))

    # Deal with outlier data

    data['Value'] = np.where(data['Value'] < data['Value'].quantile(0.10), data['Value'].quantile(0.10), data['Value'])
    data['Value'] = np.where(data['Value'] > data['Value'].quantile(0.90), data['Value'].quantile(0.90), data['Value'])
    # return data

    return data


# Function to split data into training and testing sets (5 pts)
def split_data(data):
    # Split data into training (80%) and testing (20%) sets

    taille = int(data.size*0.80)
    train_ = data[: taille]
    test_ = data[taille:]

    #train, test = train_test_split(data, test_size=0.2, random_state=42)

    return train_, test_


# Function to train a model with hyperparameters (30 pts)
def train_model(train):
    # Train a or many models with hyperparameter tuning
    # Return best model

    # prophet model

      # - reset index and rename column dataframe

    train_pro = train.reset_index()
    train_pro.rename(columns={"date": "ds", "Value": "y"}, inplace=True)
    prophet = Prophet()
    prophet.fit(train_pro)


    # Fit an ARIMA model to each time series in your data
    # You can loop through your data if you have multiple time series

    model = sm.tsa.ARIMA(train, order=(1, 0, 3))
    results = model.fit()
    print(results.summary())

    return prophet, results



# Function to evaluate the model (15 pts)
def evaluate_model(model, model2, test):
    # Evaluate the best model

    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)['yhat'][-len(test):]
    #fig = model.plot(forecast)

    mae = mean_absolute_error(test['Value'], forecast)

    forecast = model2.predict(steps=len(test))[-len(test):]
    mae1 = mean_absolute_error(test['Value'], forecast)

    print(mae, mae1)

    return mae, mae1



# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, model2, test, train):
    # Deploy the best model using Streamlit or Flask (bonus)
    # Deploy with streamlit
    st.title('Time serie model')

    st.write("My App with Streamlit, for deploy model time series")
    number = st.slider("Pick a period", 2, 365)


    st.write(f'Prévisions pour les prochaines {number} périodes :')

    st.write('ARIMA model')
    predict = model.predict(start=len(test), end=len(test) + number - 1)  #(steps=number)

    st.write(predict)

    st.subheader("Forecasted Data with ARIMA model")
    fig_forecast = px.line(predict, title='Forecasted Data')
    st.plotly_chart(fig_forecast)

    st.write('prophet model')
    future = model2.make_future_dataframe(periods=number + len(test))
    forecast = model2.predict(future)[['ds', 'yhat']][-number:]

    st.write(forecast)



    st.subheader("Forecasted Data with prophet model")
    fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Data')
    st.plotly_chart(fig_forecast)


# Main function
def main():
    # Load data
    data = load_data('malaria.xlsx', 'Feuille 1')

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Split data
    train, test = split_data(preprocessed_data)

    # Train a model with hyperparameters
    best_model1, best_model2 = train_model(train)

    # Evaluate the model
    evaluate_model(best_model1, best_model2, test)

    print()

    # Deploy the model (bonus)
    deploy_model(best_model2, best_model1, train, test)


if __name__ == "__main__":
    main()
