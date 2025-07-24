# DEPENDENCIES
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# TIME-SERIES ANALYSIS
class TimeSeriesForecaster:
    def __init__(self, df):
        self.df             = df.copy()
        self.monthly_series = self.prepare_monthly_series()

        
    def prepare_monthly_series(self):
        df            = self.df.copy()
        df["Date"]    = pd.to_datetime(df["Date"])
        monthly       = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
        monthly.index = monthly.index.to_timestamp()
            
        return monthly
    
    def arima_forecast(self, steps = 6, order = (1, 1, 1)):
        model    = ARIMA(self.monthly_series, order = order)
        results  = model.fit()
        forecast = results.forecast(steps = steps)
            
        return forecast

        
    def exponential_smoothing_forecast(self, steps = 6):
        model    = ExponentialSmoothing(self.monthly_series, 
                                        trend    = 'add', 
                                        seasonal = None,
                                       )
        results  = model.fit()
        forecast = results.forecast(steps)
            
        return forecast

        
    def holt_linear_forecast(self, steps = 6):
        model    = Holt(self.monthly_series)
        results  = model.fit()
        forecast = results.forecast(steps)
            
        return forecast

        
    def simple_avg_forecast(self, steps = 6):
        avg_value = self.monthly_series.mean()
        forecast  = [avg_value] * steps
        output    = pd.Series(forecast, 
                              index = pd.date_range(self.monthly_series.index[-1] + pd.DateOffset(months=1), 
                                                    periods = steps, 
                                                    freq    = "M",
                                                   )
                             )
        return output
            
    
    def naive_forecast(self, steps=6):
        last_value     = self.monthly_series.iloc[-1]
        forecast       = [last_value] * steps
        naive_forecast = pd.Series(forecast, 
                                   index = pd.date_range(self.monthly_series.index[-1] + pd.DateOffset(months=1), 
                                                         periods = steps, 
                                                         freq    = "M",
                                                        )
                                  )
        return naive_forecast
            
    
    def generate_plot(self, forecast_series, model_name="Forecast", steps = 6):
        date_range  = pd.date_range(self.monthly_series.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M")
        forecast_df = pd.Series(forecast_series.values, 
                                index = date_range)
    
        fig         = go.Figure()
        
        fig.add_trace(go.Scatter(x    = self.monthly_series.index, 
                                 y    = self.monthly_series.values, 
                                 name = "Historical Revenue",
                                )
                     )
            
        fig.add_trace(go.Scatter(x    = forecast_df.index, 
                                 y    = forecast_df.values, 
                                 name = f"{model_name}", 
                                 line = dict(color = "firebrick")
                                )
                     )
            
        fig.update_layout(title       = f"{steps}-Month Revenue Forecast ({model_name})", 
                          xaxis_title = "Date", 
                          yaxis_title = "Revenue",
                         )
        return fig


    def seasonal_volume_forecast(self, target_column = "Units Sold"):
        """
        Forecast and visualize seasonal volume patterns (e.g., Units Sold) across calendar months
        """
        df                      = self.df.copy()
        df["Date"]              = pd.to_datetime(df["Date"])
        df["Month"]             = df["Date"].dt.month_name()

        month_order             = ["January", 
                                   "February", 
                                   "March", 
                                   "April", 
                                   "May", 
                                   "June", 
                                   "July",
                                   "August", 
                                   "September", 
                                   "October", 
                                   "November", 
                                   "December",
                                  ]
        
        monthly_volume          = df.groupby("Month")[target_column].sum().reset_index()
        monthly_volume["Month"] = pd.Categorical(monthly_volume["Month"], 
                                                 categories = month_order, 
                                                 ordered    = True,
                                                )
                                                
        monthly_volume          = monthly_volume.sort_values("Month")

        fig                     = px.line(data_frame = monthly_volume,
                                          x          = "Month",
                                          y          = target_column,
                                          markers    = True,
                                          title      = f"📅 Monthly {target_column} Volume (Seasonal Trend)",
                                         )
                            
        return fig
