# DEPENDENCIES
import numpy as np
import pandas as pd


# RISK SUMMARY EVALUATOR
class RiskSummaryEvaluator:
    def __init__(self, filtered_df, total_df):
        self.filtered = filtered_df.copy()
        self.total    = total_df.copy()

        # Add Profit_Margin column if it doesn't exist
        if ('Profit_Margin' not in self.filtered.columns):
            self.filtered['Profit_Margin'] = (self.filtered['Profit'] / self.filtered['Revenue']) * 100
        
        if ('Profit_Margin' not in self.total.columns):
            self.total['Profit_Margin'] = (self.total['Profit'] / self.total['Revenue']) * 100

        self._prepare_metrics()

    
    def _prepare_metrics(self):
        # Overall metrics for delta comparisons
        self.total_revenue_overall     = self.total["Revenue"].sum()
        self.total_profit_overall      = self.total["Profit"].sum()
        self.avg_profit_margin_overall = self.total["Profit_Margin"].mean()

        # Revenue Volatility
        monthly_rev                    = self.filtered.groupby(self.filtered['Date'].dt.to_period('M'))['Revenue'].sum()
        self.volatility                = (monthly_rev.std() / monthly_rev.mean()) * 100 if monthly_rev.mean() != 0 else 0

        # HHI Concentration Risks
        country_share                  = self.filtered.groupby("Country")["Revenue"].sum() / self.filtered["Revenue"].sum() * 100
        product_share                  = self.filtered.groupby("Product")["Revenue"].sum() / self.filtered["Revenue"].sum() * 100
        self.hhi_country               = sum(country_share ** 2)
        self.hhi_product               = sum(product_share ** 2)

        # Low Margin %
        self.low_margin_threshold      = 30
        self.low_margin_pct            = (self.filtered["Profit_Margin"] < self.low_margin_threshold).mean() * 100

        # Performance Trend Risk
        date_75q                       = self.filtered["Date"].quantile(0.75)
        recent                         = self.filtered[self.filtered["Date"] >= date_75q]
        historical                     = self.filtered[self.filtered["Date"] < date_75q]
        self.recent_avg_revenue        = recent["Revenue"].mean()
        self.historical_avg_revenue    = historical["Revenue"].mean()

        if (self.historical_avg_revenue != 0):
            self.perf_decline = ((self.recent_avg_revenue - self.historical_avg_revenue) / self.historical_avg_revenue) * 100
        
        else:
            self.perf_decline = 0.0

    
    def generate_summary_text(self):
        return f"""
        - 📉 **Revenue Volatility (CV):** {self.volatility:.2f}%
        - 🏷️ **Market Concentration (HHI):** {self.hhi_country:.0f}
        - 🏷️ **Product Concentration (HHI):** {self.hhi_product:.0f}
        - 💸 **Low Margin Share (<{self.low_margin_threshold}%):** {self.low_margin_pct:.1f}%
        - 📉 **Performance Change (Recent vs. Historical):** {self.perf_decline:.1f}%
        """
        

    def generate_risk_insights(self):
        insights = list()

        if (self.volatility > 20):
            insights.append(f"⚠️ **High Revenue Volatility**: {self.volatility:.2f}% (consider stability strategies)")
        
        else:
            insights.append(f"✅ **Stable Revenue Volatility**: {self.volatility:.2f}%")

        
        if (self.hhi_country > 2500):
            insights.append(f"❗ **High Market Concentration Risk (HHI = {self.hhi_country:.0f})** — diversify markets")
        
        elif (self.hhi_country > 1500):
            insights.append(f"⚠️ **Moderate Market Concentration Risk (HHI = {self.hhi_country:.0f})**")
        
        else:
            insights.append(f"✅ **Low Market Concentration (HHI = {self.hhi_country:.0f})**")

        
        if (self.hhi_product > 2500):
            insights.append(f"❗ **High Product Concentration Risk (HHI = {self.hhi_product:.0f})** — diversify products")
        
        elif (self.hhi_product > 1500):
            insights.append(f"⚠️ **Moderate Product Concentration Risk (HHI = {self.hhi_product:.0f})**")
        
        else:
            insights.append(f"✅ **Low Product Concentration (HHI = {self.hhi_product:.0f})**")

        
        if (self.low_margin_pct > 50):
            insights.append(f"❗ **Over {self.low_margin_pct:.1f}% transactions are low-margin (<{self.low_margin_threshold}%)** — profitability risk")
        
        else:
            insights.append(f"✅ **Healthy Profit Margins, only {self.low_margin_pct:.1f}% low-margin transactions**")

        
        if (self.perf_decline < 0):
            insights.append(f"❗ **Recent revenue declined by {self.perf_decline:.1f}% compared to historical** — recent performance dip")
        
        else:
            insights.append(f"✅ **Recent revenue improved by {self.perf_decline:.1f}%**")

        return insights

    
    def detect_low_margin_transactions(self, threshold = None):
        """
        Detect and analyze low margin transactions with detailed breakdown
        
        Arguments:
        ----------
            threshold { float } : Custom threshold for low margin detection. 
                                  If None, uses the class default (30%)
        
        Returns:
        --------
                { dict }        : Dictionary containing low margin analysis results
        """
        if (threshold is None):
            threshold = self.low_margin_threshold
        
        # Identify low margin transactions
        low_margin_mask         = self.filtered['Profit_Margin'] < threshold
        low_margin_transactions = self.filtered[low_margin_mask].copy()
        
        # Calculate statistics
        total_transactions      = len(self.filtered)
        low_margin_count        = len(low_margin_transactions)
        low_margin_percentage   = (low_margin_count / total_transactions) * 100
        
        # Revenue and profit impact
        low_margin_revenue      = low_margin_transactions['Revenue'].sum()
        low_margin_profit       = low_margin_transactions['Profit'].sum()
        total_revenue           = self.filtered['Revenue'].sum()
        total_profit            = self.filtered['Profit'].sum()
        
        revenue_impact          = (low_margin_revenue / total_revenue) * 100
        profit_impact           = (low_margin_profit / total_profit) * 100
        
        # Breakdown by country and product
        country_breakdown       = low_margin_transactions.groupby('Country').agg({'Revenue'       : 'sum',
                                                                                  'Profit'        : 'sum',
                                                                                  'Profit_Margin' : 'mean',
                                                                                }).round(2)
        
        product_breakdown       = low_margin_transactions.groupby('Product').agg({'Revenue'       : 'sum',
                                                                                  'Profit'        : 'sum',
                                                                                  'Profit_Margin' : 'mean',
                                                                                }).round(2)
        
        # Average margin of low margin transactions
        avg_low_margin          = low_margin_transactions['Profit_Margin'].mean()
        
        return {'threshold'               : threshold,
                'total_transactions'      : total_transactions,
                'low_margin_count'        : low_margin_count,
                'low_margin_percentage'   : round(low_margin_percentage, 2),
                'low_margin_revenue'      : low_margin_revenue,
                'low_margin_profit'       : low_margin_profit,
                'revenue_impact_pct'      : round(revenue_impact, 2),
                'profit_impact_pct'       : round(profit_impact, 2),
                'avg_low_margin'          : round(avg_low_margin, 2),
                'country_breakdown'       : country_breakdown,
                'product_breakdown'       : product_breakdown,
                'low_margin_transactions' : low_margin_transactions,
               }


    def calculate_return_risk(self, risk_free_rate=0.03):
        """
        Calculate return risk metrics including ROI volatility and risk-adjusted returns
        
        Arguments:
        ----------
            risk_free_rate { float } : Risk-free rate for Sharpe ratio calculation (default 3%)
        
        Returns:
        --------
                 { dict }            : Dictionary containing return risk analysis
        """
        # Calculate ROI for each transaction
        self.filtered['ROI']          = (self.filtered['Profit'] / self.filtered['Cost']) * 100
        
        # Monthly aggregated returns
        monthly_data                  = self.filtered.groupby(self.filtered['Date'].dt.to_period('M')).agg({'Revenue' : 'sum',
                                                                                                            'Profit'  : 'sum',
                                                                                                            'Cost'    : 'sum',
                                                                                                          })
        
        monthly_data['ROI']           = (monthly_data['Profit'] / monthly_data['Cost']) * 100
        monthly_data['Profit_Margin'] = (monthly_data['Profit'] / monthly_data['Revenue']) * 100
        
        # Risk metrics
        roi_mean                      = monthly_data['ROI'].mean()
        roi_std                       = monthly_data['ROI'].std()
        roi_cv                        = (roi_std / roi_mean) * 100 if roi_mean != 0 else 0
        
        # Sharpe ratio (assuming monthly data)
        excess_return                 = roi_mean - (risk_free_rate * 100 / 12)  # Convert annual to monthly
        sharpe_ratio                  = excess_return / roi_std if roi_std != 0 else 0
        
        # Value at Risk (VaR) - 5% worst case
        var_5                         = np.percentile(monthly_data['ROI'], 5)
        
        # Maximum drawdown
        cumulative_returns            = (1 + monthly_data['ROI']/100).cumprod()
        running_max                   = cumulative_returns.expanding().max()
        drawdown                      = (cumulative_returns / running_max) - 1
        max_drawdown                  = drawdown.min() * 100
        
        # Downside deviation (volatility of negative returns only)
        negative_returns              = monthly_data['ROI'][monthly_data['ROI'] < roi_mean]
        downside_deviation            = negative_returns.std() if len(negative_returns) > 0 else 0
        
        return {'avg_monthly_roi'           : round(roi_mean, 2),
                'roi_volatility'            : round(roi_std, 2),
                'roi_coefficient_variation' : round(roi_cv, 2),
                'sharpe_ratio'              : round(sharpe_ratio, 2),
                'value_at_risk_5pct'        : round(var_5, 2),
                'max_drawdown_pct'          : round(max_drawdown, 2),
                'downside_deviation'        : round(downside_deviation, 2),
                'monthly_roi_data'          : monthly_data['ROI'].tolist(),
                'risk_assessment'           : self._assess_return_risk(roi_cv, sharpe_ratio, max_drawdown),
               }

    
    def _assess_return_risk(self, cv, sharpe, max_dd):
        """
        Helper method to assess overall return risk level
        """
        risk_score = 0
        
        # Coefficient of variation scoring
        if (cv > 50):
            risk_score += 3
        
        elif (cv > 30):
            risk_score += 2
        
        elif (cv > 15):
            risk_score += 1
        
        # Sharpe ratio scoring (lower is riskier)
        if (sharpe < 0):
            risk_score += 3
        
        elif (sharpe < 0.5):
            risk_score += 2
        
        elif (sharpe < 1):
            risk_score += 1
        
        # Max drawdown scoring
        if (abs(max_dd) > 30):
            risk_score += 3
        
        elif (abs(max_dd) > 15):
            risk_score += 2

        elif (abs(max_dd) > 5):
            risk_score += 1
        
        
        # Risk Brackets
        if (risk_score >= 7):
            return "HIGH RISK"

        elif (risk_score >= 4):
            return "MODERATE RISK"

        else:
            return "LOW RISK"

    
    def profitability_trend_alert(self, periods=4, alert_threshold=-10):
        """
        Analyze profitability trends and generate alerts for declining performance
        
        Arguments:
        ----------
            periods          { int } : Number of periods to analyze for trend (default 4)

            alert_threshold { float } : Threshold for decline alert in % (default -10%)
        
        Returns:
        --------
                   { dict }           : Dictionary containing profitability trend analysis and alerts
        """
        # Create monthly periods for trend analysis
        monthly_data                  = self.filtered.groupby(self.filtered['Date'].dt.to_period('M')).agg({'Revenue' : 'sum',
                                                                                                            'Profit'  : 'sum',
                                                                                                            'Cost'    : 'sum',
                                                                                                          }).reset_index()
        
        monthly_data['Profit_Margin'] = (monthly_data['Profit'] / monthly_data['Revenue']) * 100
        monthly_data['ROI']           = (monthly_data['Profit'] / monthly_data['Cost']) * 100
        
        # Sort by date to ensure proper trend analysis
        monthly_data                  = monthly_data.sort_values('Date')
        
        alerts                        = list()
        trends                        = dict()
        
        if (len(monthly_data) < periods):
            return {'alerts'       : ["⚠️ Insufficient data for trend analysis"],
                    'trends'       : {},
                    'monthly_data' : monthly_data,
                   }
        
        # Analyze recent periods vs previous periods
        recent_periods                = monthly_data.tail(periods)
        previous_periods              = monthly_data.iloc[-(periods*2):-periods] if len(monthly_data) >= periods*2 else monthly_data.iloc[:-periods]
        
        # Revenue trend
        recent_avg_revenue            = recent_periods['Revenue'].mean()
        previous_avg_revenue          = previous_periods['Revenue'].mean()
        revenue_change                = ((recent_avg_revenue - previous_avg_revenue) / previous_avg_revenue) * 100 if previous_avg_revenue != 0 else 0
        
        trends['revenue_change_pct']  = round(revenue_change, 2)
        
        if (revenue_change < alert_threshold):
            alerts.append(f"🚨 **REVENUE ALERT**: {revenue_change:.1f}% decline over last {periods} periods")
        
        # Profit margin trend
        recent_avg_margin             = recent_periods['Profit_Margin'].mean()
        previous_avg_margin           = previous_periods['Profit_Margin'].mean()
        margin_change                 = (recent_avg_margin - previous_avg_margin)
        
        trends['margin_change_pts']   = round(margin_change, 2)
        
        # More sensitive for margins
        if (margin_change < (alert_threshold / 2)): 
            alerts.append(f"🚨 **MARGIN ALERT**: {margin_change:.1f} percentage points decline in profit margin")
        
        # ROI trend
        recent_avg_roi                = recent_periods['ROI'].mean()
        previous_avg_roi              = previous_periods['ROI'].mean()
        roi_change                    = ((recent_avg_roi - previous_avg_roi) / previous_avg_roi) * 100 if previous_avg_roi != 0 else 0
        
        trends['roi_change_pct']      = round(roi_change, 2)
        
        if (roi_change < alert_threshold):
            alerts.append(f"🚨 **ROI ALERT**: {roi_change:.1f}% decline in return on investment")
        
        # Consecutive declining periods check
        revenue_declining = 0
        margin_declining  = 0
        
        for i in range(1, len(recent_periods)):
            if (recent_periods.iloc[i]['Revenue'] < recent_periods.iloc[i-1]['Revenue']):
                revenue_declining += 1
            
            if (recent_periods.iloc[i]['Profit_Margin'] < recent_periods.iloc[i-1]['Profit_Margin']):
                margin_declining += 1
        
        if (revenue_declining >= periods - 1):
            alerts.append(f"🚨 **CONSECUTIVE DECLINE ALERT**: Revenue declining for {revenue_declining} consecutive periods")
        
        if (margin_declining >= periods - 1):
            alerts.append(f"🚨 **CONSECUTIVE DECLINE ALERT**: Profit margin declining for {margin_declining} consecutive periods")
        
        # Trend direction analysis
        revenue_trend     = "DECLINING" if revenue_change < -5 else "STABLE" if revenue_change < 5 else "GROWING"
        margin_trend      = "DECLINING" if margin_change < -2 else "STABLE" if margin_change < 2 else "IMPROVING"
        roi_trend         = "DECLINING" if roi_change < -5 else "STABLE" if roi_change < 5 else "IMPROVING"
        
        trends.update({'revenue_trend'      : revenue_trend,
                       'margin_trend'       : margin_trend,
                       'roi_trend'          : roi_trend,
                       'recent_avg_revenue' : round(recent_avg_revenue, 2),
                       'recent_avg_margin'  : round(recent_avg_margin, 2),
                       'recent_avg_roi'     : round(recent_avg_roi, 2),
                       'periods_analyzed'   : periods,
                     })
        
        # Success messages if no alerts
        if not alerts:
            alerts.append("✅ **All profitability trends are healthy**")
        
        return {'alerts'       : alerts,
                'trends'       : trends,
                'monthly_data' : monthly_data,
               }


    