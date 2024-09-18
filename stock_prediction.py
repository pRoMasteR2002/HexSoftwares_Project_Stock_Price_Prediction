# Required libraries and installation commands:
# Run the following commands in your terminal to install the required libraries:
#
# pip install dash (For building the web application.)
# pip install plotly (For creating interactive graphs.)
# pip install pandas (For data manipulation and handling.)
# pip install numpy (For numerical operations.)
# pip install scikit-learn (For machine learning models.)
# pip install yfinance (For fetching stock data.)
# pip install dash-core-components (Essential components for Dash apps (though it’s included in Dash, you may install it explicitly).)
# pip install dash-html-components (For creating HTML elements in Dash.)

# Import necessary libraries
import dash
from dash import dcc, html  # dcc for interactive components, html for HTML tags
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go  # For creating interactive plots
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf  # For fetching stock data

def generate_stock_data(stock_code):
    
    #Fetch historical stock data for a given stock code.
    
    #Args:
    #stock_code (str): The stock symbol (e.g., 'AAPL' for Apple Inc.)
    
    #Returns:
    #pandas.DataFrame: A dataframe with 'Date' and 'Value' (closing price) columns
    
    stock = yf.Ticker(stock_code)
    df = stock.history(start="2020-01-01", end="2028-01-01")
    return df.reset_index()[['Date', 'Close']].rename(columns={'Close': 'Value'})

def add_technical_indicators(df):
    
    #Add technical indicators (SMA20, SMA50, RSI) to the stock data.
    
    #Args:
    #df (pandas.DataFrame): Dataframe with 'Date' and 'Value' columns
    
    #Returns:
    #pandas.DataFrame: Dataframe with added technical indicators
    
    # Calculate 20-day and 50-day Simple Moving Averages
    
    df['SMA_20'] = df['Value'].rolling(window=20).mean()
    df['SMA_50'] = df['Value'].rolling(window=50).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['Value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def train_model(df):
    
    #Train a linear regression model on the stock data.
    
    #Args:
    #df (pandas.DataFrame): Dataframe with stock data and technical indicators
    
    #Returns:
    #tuple: (trained model, mean squared error) or (None, error message)
    
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    
    if len(df) < 7:  # Ensure there's enough data of at least a week for training
        return None, "Not enough data to train the model."

    X = df[['SMA_20', 'SMA_50', 'RSI']]
    y = df['Value']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return model, mse

def predict_stock_prices(model, df):
    
    #Predict stock prices using the trained model.
    
    #Args:
    #model: Trained LinearRegression model
    #df (pandas.DataFrame): Dataframe with stock data
    
    #Returns:
    #tuple: (DataFrame with predictions, error message if any)
    
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    
    if model is None:
        return df, "Model could not be trained due to insufficient data."
    
    X = df[['SMA_20', 'SMA_50', 'RSI']]
    df['Predicted_Value'] = model.predict(X)
    
    return df, ""

def generate_remarks(df):
    
    #Generate remarks based on the predicted stock performance.
    
    #Args:
    #df (pandas.DataFrame): Dataframe with actual and predicted stock values
    
    #Returns:
    #str: A remark about the stock's predicted performance
    
    if df.empty or 'Predicted_Value' not in df.columns:
        return "No predictions available."

    initial_value = df['Value'].iloc[0]
    final_value = df['Predicted_Value'].iloc[-1]
    percentage_change = ((final_value - initial_value) / initial_value) * 100

    if percentage_change > 5:
        remarks = f"The stock is performing well with a {percentage_change:.2f}% predicted increase."
    elif percentage_change < -5:
        remarks = f"The stock performance is predicted to decline by {-percentage_change:.2f}%. Consider revisiting investment strategies."
    else:
        remarks = f"The stock is expected to have a moderate change of {percentage_change:.2f}%."
    
    return remarks

# Dictionary of predefined stock data (10 different tech giants)
stocks_data = {
    'AAPL': 'Apple Technologies',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix, Inc.',
    'ADBE': 'Adobe Inc.',
    'INTC': 'Intel Corporation'
}

# Create Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # Container for dropdown and show button
    html.Div([
        # Stock dropdown
        html.Div([
            dcc.Dropdown(
                id='stock-dropdown',
                options=[{'label': f'{name} ({code})', 'value': code} for code, name in stocks_data.items()],
                value='AAPL',  # Default value
                clearable=False,
                style={'font-size': '18px', 'font-weight': 'bold'}
            ),
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'middle'}),

        # Predict/Show button
        html.Div([
            html.Button('Show', id='predict-btn', n_clicks=0, 
                        style={'font-size': '18px', 'font-weight': 'bold', 'padding': '10px 20px', 'height': '38px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle', 'textAlign': 'center'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'paddingBottom': '20px'}),
    
    # Stock graph
    dcc.Graph(id='stock-graph'),

    # Date range display
    html.Div(id='date-range-display', style={'font-size': '18px', 'font-weight': 'bold', 'padding-top': '10px'}),

    # Stock values display
    html.Div(id='predicted-stock-values', 
             style={'font-size': '18px', 'padding-top': '20px', 'line-height': '1.5'}),

], style={'font-family': 'Arial, sans-serif', 'padding': '20px'})

# Callback to update the stock graph based on selection
@app.callback(
    Output('stock-graph', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('date-range-display', 'children')]
)
def update_graph(selected_stock, date_range):
    
    #Update the stock graph based on the selected stock and date range.
    
    #Args:
    #selected_stock (str): The selected stock code
    #date_range (str): The selected date range
    
    #Returns:
    #dict: A dictionary describing the updated graph figure
    
    df_stock = generate_stock_data(selected_stock)  # Generate data for the selected stock
    
    # Parse date range if available
    start_date = None
    end_date = None
    if date_range and 'Start Date' in date_range:
        parts = date_range.split(", End Date: ")
        start_date = pd.to_datetime(parts[0].split(": ")[1]).date()
        if len(parts) > 1:
            end_date = pd.to_datetime(parts[1]).date()
        else:
            end_date = df_stock['Date'].max().date()  # Use the last available date if end date is not specified

    # Filter data based on selected date range
    if start_date and end_date:
        df_stock = df_stock[(df_stock['Date'].dt.date >= start_date) & (df_stock['Date'].dt.date <= end_date)]

    # Create the trace for the stock data
    trace = go.Scatter(
        x=df_stock['Date'],
        y=df_stock['Value'],
        mode='lines+markers',
        marker={'color': ['green' if val > 1000 else 'red' for val in df_stock['Value']]},
        name=f'Stock {selected_stock}'
    )

    # Add highlighting for selected date range
    shapes = []
    if start_date and end_date:
        shapes.append({
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': start_date,
            'y0': 0,
            'x1': end_date,
            'y1': 1,
            'fillcolor': 'rgba(0,0,0,0.1)',
            'line': {'width': 0},
        })

    # Return the figure dictionary
    return {
        'data': [trace],
        'layout': go.Layout(
            title={
                'text': f'{stocks_data.get(selected_stock, selected_stock)} Stock Price',
                'font': {'size': 18, 'family': 'Arial, sans-serif', 'weight': 'bold'}
            },
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            hovermode='closest',
            shapes=shapes
        )
    }

# Callback to display date range on graph click
@app.callback(
    Output('date-range-display', 'children'),
    [Input('stock-graph', 'clickData')],
    [State('date-range-display', 'children')]
)
def display_selected_dates(click_data, current_display):
    
    #Update the displayed date range when the user clicks on the graph.
    
    #Args:
    #click_data (dict): Data about where the user clicked on the graph
    #current_display (str): The current date range display
    
    #Returns:
    #str: Updated date range display
    
    if click_data is None:
        return current_display
    else:
        date_clicked = pd.to_datetime(click_data['points'][0]['x']).date()
        
        if current_display and 'Start Date' in current_display:
            start_date = pd.to_datetime(current_display.split(": ")[1].split(',')[0]).date()
            if date_clicked < start_date:
                start_date, date_clicked = date_clicked, start_date
            return f'Start Date: {start_date}, End Date: {date_clicked}'
        else:
            return f'Start Date: {date_clicked}'

# Callback to display stock values between selected dates and show remarks
@app.callback(
    Output('predicted-stock-values', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('stock-dropdown', 'value'), State('date-range-display', 'children')]
)
def show_stock_values(n_clicks, selected_stock, date_range):
    
    #Display predicted stock values and remarks when the user clicks the 'Show' button.
    
    #Args:
    #n_clicks (int): Number of times the 'Show' button has been clicked
    #selected_stock (str): The selected stock code
    #date_range (str): The selected date range
    
    #Returns:
    #list: HTML components displaying predicted stock values and remarks
    
    if n_clicks > 0 and date_range:
        try:
            # Extract dates from date range display
            if 'Start Date' in date_range:
                parts = date_range.split(", End Date: ")
                start_date = pd.to_datetime(parts[0].split(": ")[1]).date()
                end_date = pd.to_datetime(parts[1]).date()
            else:
                start_date = pd.to_datetime(date_range.split(": ")[1]).date()
                end_date = start_date

            # Generate stock data for the selected stock
            df_stock = generate_stock_data(selected_stock)
            df_stock['Date'] = df_stock['Date'].dt.date
            df_stock = df_stock[(df_stock['Date'] >= start_date) & (df_stock['Date'] <= end_date)]
            
            if df_stock.empty:
                return f"No stock data available between {start_date} and {end_date}."
            
            # Train the model and make predictions
            model, message = train_model(df_stock)
            if model is None:
                return f"Error: {message}"
            
            df_stock, prediction_message = predict_stock_prices(model, df_stock)
            if prediction_message:
                return f"Error: {prediction_message}"
            
            # Generate remarks
            remarks = generate_remarks(df_stock)

            # Find the minimum and maximum stock values for color scaling
            min_value = df_stock['Predicted_Value'].min()
            max_value = df_stock['Predicted_Value'].max()

            # Return the predicted values and remarks as HTML components
            return [
                html.P(f"Displaying predicted stock values for {selected_stock} from {start_date} to {end_date}:", style={'font-weight': 'bold'}),
                html.Ul([
                    html.Li(
                        f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Predicted_Value']:.2f}",
                        style={'color': 'red' if row['Predicted_Value'] < min_value + (max_value - min_value) / 2 else 'green'}
                    ) for _, row in df_stock.iterrows()
                ]),
                html.P(f"Remarks: {remarks}", style={'color': 'green' if 'well' in remarks else 'red', 'font-size': '18px', 'font-weight': 'bold'})
            ]
        except Exception as e:
            return f"Error processing date range: {e}"
    
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)