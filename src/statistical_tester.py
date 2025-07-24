# DEPENDENCIES
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import normaltest


# STATISTICAL TESTS
class StatisticalTester:
    def __init__(self, df):
        self.df = df.copy()

    
    def get_grouped_data(self, metric, group_by):
        grouped_data = [group[metric].dropna().values for _, group in self.df.groupby(group_by)]
        
        return grouped_data

    
    def shapiro_test(self, metric):
        values  = self.df[metric].dropna().values
        stat, p = stats.shapiro(values)
        
        return stat, p

    
    def anova_test(self, grouped_data):
        f_val, p_val = stats.f_oneway(*grouped_data)
        
        return f_val, p_val

    
    def levene_test(self, grouped_data):
        stat, p = stats.levene(*grouped_data)
        
        return stat, p

    
    def kruskal_test(self, grouped_data):
        h_val, p_val = stats.kruskal(*grouped_data)
        
        return h_val, p_val

    
    def mann_whitney_test(self, group1, group2):
        u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative = 'two-sided')
        
        return u_stat, p_val


    def boxplot_figure(self, metric, group_by):
        fig = px.box(data_frame = self.df, 
                     x          = group_by, 
                     y          = metric, 
                     color      = group_by,
                     title      = f"{metric} Distribution by {group_by}")
        
        return fig
        
     
    def estimate_price_elasticity(self, price_col = "Price", quantity_col = "Quantity"):
        # Drop NaNs and ensure positive values for log
        df_valid                 = self.df[[price_col, quantity_col]].dropna()
        df_valid                 = df_valid[(df_valid[price_col] > 0) & (df_valid[quantity_col] > 0)].copy()

        # Log transformation
        df_valid["log_price"]    = np.log(df_valid[price_col])
        df_valid["log_quantity"] = np.log(df_valid[quantity_col])

        # Linear regression: log(Q) ~ log(P)
        X                        = sm.add_constant(df_valid["log_price"])
        y                        = df_valid["log_quantity"]
        model                    = sm.OLS(y, X).fit()

        price_elasticity         = model.params["log_price"]

        # Plotting
        fig                      = px.scatter(data_frame = df_valid,
                                              x          = "log_price",
                                              y          = "log_quantity",
                                              trendline  = "ols",
                                              title      = f"Log-Log Regression: Quantity vs Price (Elasticity = {price_elasticity:.2f})",
                                             )

        model_summary            = model.summary()

        return price_elasticity, model_summary, fig
