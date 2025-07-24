# DEPENDENCIES
import numpy as np
import pandas as pd


# MARKET SEGMENTATION
class MarketSegmentationSummary:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    
    def calculate_market_metrics(self):
        metrics                          = (self.df.groupby("Country").agg({"Revenue"       : ["sum", "mean", "count"],
                                                                            "Profit"        : ["sum", "mean"],
                                                                            "Units Sold"    : ["sum", "mean"],
                                                                            "Profit_Margin" : "mean",
                                                                          }).round(2)
                                           )
        
        metrics.columns                  = ["Total_Revenue", 
                                            "Avg_Revenue", 
                                            "Transactions",
                                            "Total_Profit", 
                                            "Avg_Profit", 
                                            "Total_Units", 
                                            "Avg_Units",
                                            "Avg_Profit_Margin",
                                           ]
        
        metrics                          = metrics.reset_index()

        # Additional Insights
        total_revenue                     = metrics["Total_Revenue"].sum()
        
        metrics["%Revenue_Contribution"]  = round(metrics["Total_Revenue"] / total_revenue * 100, 2)
        metrics["Revenue_per_Unit"]       = (metrics["Total_Revenue"] / metrics["Total_Units"]).round(2)
        metrics["Profit_per_Transaction"] = (metrics["Total_Profit"] / metrics["Transactions"]).round(2)
        metrics["Country_Rank_Revenue"]   = metrics["Total_Revenue"].rank(ascending=False).astype(int)

        output_dict                       = {"market_table"         : metrics,
                                             "top3_by_revenue"      : metrics.nlargest(3, "Total_Revenue"),
                                             "country_wise_revenue" : metrics[["Country", "Total_Revenue"]],
                                             "pie_data"             : metrics[["Country", "%Revenue_Contribution"]],
                                            }
        
        return output_dict

    
    def generate_product_country_affinity_matrix(self):
        """
        Create a matrix of revenue contribution by product across countries
        """
        pivot           = self.df.pivot_table(index      = "Product",
                                              columns    = "Country",
                                              values     = "Revenue",
                                              aggfunc    = "sum",
                                              fill_value = 0,
                                             )

        affinity_matrix = pivot.div(pivot.sum(axis = 0), 
                                    axis = 1).round(3)

        return affinity_matrix.reset_index()


    def country_value_segmentation(self, quantiles: tuple = (0.33, 0.66)):
        """
        Segment countries into Low / Medium / High based on revenue thresholds
        """
        country_revenue = self.df.groupby("Country")["Revenue"].sum().reset_index()
        q_low, q_high   = country_revenue["Revenue"].quantile(quantiles)

        def label_segment(val):
            if (val <= q_low):
                return "Low Value"
            
            elif (val <= q_high):
                return "Mid Value"
            
            else:
                return "High Value"

        country_revenue["Value_Segment"] = country_revenue["Revenue"].apply(label_segment)
        sorted_country_revenue           = country_revenue.sort_values(by        = "Revenue", 
                                                                       ascending = False,
                                                                      )

        return sorted_country_revenue
