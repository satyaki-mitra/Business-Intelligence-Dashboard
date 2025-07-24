# DEPENDENCIES
import pandas as pd


# BUSINESS RISK EVALUATOR
class BusinessRiskEvaluator:
    def __init__(self, filtered_df: pd.DataFrame):
        """
        Class to compute business health alerts and risk KPIs from filtered data
        """
        self.filtered = filtered_df.copy()
        self._prepare_monthly_revenue()

    
    def _prepare_monthly_revenue(self):
        # Monthly revenue preparation
        monthly              = self.filtered.groupby(self.filtered['Date'].dt.to_period('M'))['Revenue'].sum()
        self.monthly_revenue = monthly.to_timestamp()

    
    def compute_latest_monthly_change(self):
        if (len(self.monthly_revenue) >= 2):
            latest, previous = self.monthly_revenue.iloc[-1], self.monthly_revenue.iloc[-2]
            change_pct       = ((latest - previous) / previous) * 100
            
            return latest, change_pct
        
        else:
            return None, None

    
    def compute_market_concentration_hhi(self):
        market_pct               = (self.filtered.groupby("Country")["Revenue"].sum() / self.filtered["Revenue"].sum()) * 100
        market_concentration_hhi = (market_pct ** 2).sum()
        
        return market_concentration_hhi

    
    def compute_low_margin_pct(self, threshold = 30):
        low_margin_pct = (self.filtered["Profit_Margin"] < threshold).mean() * 100
        
        return low_margin_pct

    
    def compute_revenue_volatility_cv(self):
        monthly               = self.monthly_revenue
        revenue_volatility_cv = (monthly.std() / monthly.mean()) * 100 if len(monthly) >= 2 else None
        
        return revenue_volatility_cv

    def calculate_support_cost_allocation(self):
        """
        Calculates average cost per unit by product to inform customer support cost allocation and
        Returns a DataFrame with Product and Cost_per_Unit
        """
        df                  = self.filtered.copy()
        df["Cost_per_Unit"] = df["Cost"] / df["Units Sold"]
        
        product_costs       = df.groupby("Product")["Cost_per_Unit"].mean().reset_index()
        product_costs       = product_costs.sort_values(by        = "Cost_per_Unit", 
                                                        ascending = False,
                                                       )
        
        return product_costs
