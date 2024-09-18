
---

# Stock Price Prediction with Dash and Plotly

Welcome to the Stock Price Prediction project! This application uses machine learning models to predict future stock prices based on historical data. Built with Dash and Plotly, this web application allows users to visualize stock price trends and make predictions using various technical indicators.

## What This Project Is

This project is a web application designed to forecast stock prices and analyze their trends. It leverages historical stock data, technical indicators, and machine learning techniques to provide predictions and insights. The application is built using Dash, a framework for creating interactive web applications in Python, and Plotly, a library for generating interactive graphs.

## What You Can Expect

- **Interactive Visualization**: Explore stock price data through interactive graphs. Customize your view by selecting different stocks and date ranges.
- **Technical Indicators**: See key technical indicators like Simple Moving Averages (SMA20, SMA50) and Relative Strength Index (RSI) calculated and displayed for better market analysis.
- **Predictive Modeling**: Use a linear regression model to predict future stock prices based on historical data and technical indicators.
- **Custom Date Range Selection**: Filter stock data and predictions based on specific date ranges you choose.
- **Dynamic Remarks**: Receive automated insights and remarks about the predicted stock performance, including potential investment strategies.

## Concepts Used and Their Sources

### 1. **Data Collection**
   - **Concept**: Fetching historical stock data.
   - **Source**: `yfinance` library.
   - **Usage**: Retrieves historical stock data such as closing prices and dates for analysis.

### 2. **Feature Engineering**
   - **Concept**: Calculating technical indicators.
   - **Source**: Financial market analysis.
   - **Usage**: 
     - **Simple Moving Averages (SMA)**: SMA20 and SMA50 are used to smooth out price data and identify trends.
     - **Relative Strength Index (RSI)**: Measures the speed and change of price movements to identify overbought or oversold conditions.

### 3. **Machine Learning Modeling**
   - **Concept**: Predictive modeling using linear regression.
   - **Source**: `scikit-learn` library.
   - **Usage**: Trains a linear regression model to predict future stock prices based on historical data and technical indicators.

### 4. **Data Visualization**
   - **Concept**: Creating interactive graphs and visualizations.
   - **Source**: `plotly` library.
   - **Usage**: Generates interactive graphs for visualizing stock price data and predictions.

### 5. **Web Application Development**
   - **Concept**: Building interactive web applications.
   - **Source**: `dash` library.
   - **Usage**: Provides an interface for users to interact with the stock price data and predictions, including dropdowns, buttons, and graphs.

## Features

- **Interactive Stock Visualization**: Visualize historical stock prices with interactive graphs.
- **Technical Indicators**: Automatically calculates and displays key technical indicators like SMA20, SMA50, and RSI.
- **Stock Predictions**: Predict future stock prices using a linear regression model.
- **Custom Date Range**: Select specific date ranges to filter data and view predictions.
- **Dynamic Remarks**: Get actionable insights based on predicted stock performance.

## Requirements

To run this project, you need to install the following Python libraries:

- `dash`: For building the web application.
- `plotly`: For creating interactive graphs.
- `pandas`: For data manipulation and handling.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning models.
- `yfinance`: For fetching stock data.
- `dash-core-components`: Essential components for Dash apps.
- `dash-html-components`: For creating HTML elements in Dash.

You can install these libraries using the following commands:

```bash
pip install dash
pip install plotly
pip install pandas
pip install numpy
pip install scikit-learn
pip install yfinance
pip install dash-core-components
pip install dash-html-components
```

## How It Works

1. **Data Collection**: Historical stock data is fetched using the `yfinance` library.
2. **Feature Engineering**: Technical indicators such as SMA20, SMA50, and RSI are calculated.
3. **Model Training**: A linear regression model is trained on the historical stock data.
4. **Prediction**: The trained model predicts future stock prices based on the selected date range.
5. **Visualization**: The application displays interactive graphs with the stock's historical data and predictions.

## Usage Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/pRoMasteR2002/HexSoftwares_Project_Stock_Price_Prediction.git
   ```

2. **Access the Application**:

   Open your web browser and navigate to `http://127.0.0.1:8050/` to interact with the application.

## Repository Link

You can access the GitHub repository here: [Stock Price Prediction GitHub Repository](https://github.com/pRoMasteR2002/HexSoftwares_Project_Stock_Price_Prediction)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to make any additional changes or ask if you need more details added!
