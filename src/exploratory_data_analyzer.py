# DEPENDENCIES
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose


# EXPLORATORY DATA ANALYSIS
class SalesEDA:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the SalesEDA class with a cleaned dataframe
        """
        self.df = dataframe.copy()

    
    def data_filtering(self):
        """
        Interactive filters for Date, Country, Product using Streamlit UI
        and returns filtered dataframe
        """
        st.subheader("🎛️ Interactive Filters")

        col1, col2, col3 = st.columns(spec = [2, 3, 3])

        with col1:
            date_range = st.date_input(label = "Date Range",
                                       value = (self.df["Date"].min(), self.df["Date"].max())
                                      )

        with col2:
            st.markdown(body = "##### Select Countries")
            countries_available = sorted(self.df["Country"].unique())
            selected_countries  = [country for country in countries_available if st.checkbox(country, True, key = f"country_{country}")]

        with col3:
            st.markdown(body = "##### Select Products")
            products_available = sorted(self.df["Product"].unique())
            selected_products = [product for product in products_available if st.checkbox(product, True, key = f"product_{product}")]

        # After getting all user inputs, filter the dataframe applying these filters
        filtered = self.df[(self.df["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
                           (self.df["Country"].isin(selected_countries)) &
                           (self.df["Product"].isin(selected_products))
                          ]

        st.info(f"Showing {len(filtered)} records after applying filters")

        return filtered

    
    def market_kpis(self, filtered_df):
        metrics                            = filtered_df.groupby("Country").agg({"Revenue"       : ["sum", "mean", "count"],
                                                                                 "Profit"        : ["sum", "mean"],
                                                                                 "Units Sold"    : ["sum", "mean"],
                                                                                 "Profit_Margin" : "mean",
                                                                               }).round(2)

        metrics.columns                    = ["Total_Revenue", 
                                              "Avg_Revenue", 
                                              "Transactions", 
                                              "Total_Profit", 
                                              "Avg_Profit", 
                                              "Total_Units", 
                                              "Avg_Units",
                                              "Avg_Profit_Margin",
                                             ]
        
        metrics                            = metrics.reset_index()

        total_revenue                      = metrics["Total_Revenue"].sum()

        metrics["%Revenue_Share_Filtered"] = (metrics["Total_Revenue"] / total_revenue * 100).round(2)
        metrics["ARPU"]                    = (metrics["Total_Revenue"] / metrics["Total_Units"]).replace([np.inf, -np.inf], 0).round(2)
        metrics["Profit_per_Transaction"]  = (metrics["Total_Profit"] / metrics["Transactions"]).replace([np.inf, -np.inf], 0).round(2)

        metrics                            = metrics.sort_values(by        = "Total_Revenue", 
                                                                 ascending = False).reset_index(drop = True)
        
        metrics["Cumulative_Revenue_%"]    = (metrics["Total_Revenue"].cumsum() / total_revenue * 100).round(2)

        # Concentration ratio (Top 20% countries by count)
        top_n                              = max(int(np.ceil(0.2 * len(metrics))), 1)
        top_n_revenue                      = metrics.loc[:top_n - 1, "Total_Revenue"].sum()
        filtered_concentration             = round((top_n_revenue / total_revenue) * 100, 2)

        # PLOTS
        pie_chart                          = px.pie(data_frame              = metrics,
                                                    names                   = "Country",
                                                    values                  = "%Revenue_Share_Filtered",
                                                    title                   = "% Revenue Share by Country (Filtered)",
                                                    color_discrete_sequence = px.colors.sequential.RdBu,
                                                    width                   = 600,
                                                    height                  = 600,
                                                   )

        bar_chart                          = px.bar(data_frame             = metrics,
                                                    x                      = "Country",
                                                    y                      = "Total_Revenue",
                                                    text                   = "%Revenue_Share_Filtered",
                                                    title                  = "Total Revenue by Country (Filtered)",
                                                    color                  = "Total_Revenue",
                                                    color_continuous_scale = "Blues",
                                                   )
        
        bar_chart.update_traces(texttemplate = '%{text:.2s}%', 
                                textposition = 'outside',
                               )

        lorenz_curve                      = px.line(data_frame = metrics,
                                                    x          = metrics.index + 1,
                                                    y          = "Cumulative_Revenue_%",
                                                    title      = "Cumulative Revenue Distribution (Lorenz Curve)",
                                                    markers    = True,
                                                   )
        
        lorenz_curve.update_layout(xaxis_title = "Country Rank", 
                                   yaxis_title = "Cumulative Revenue %",
                                  )

        output_dict                      = {"filtered_market_table"        : metrics,
                                            "filtered_concentration_ratio" : filtered_concentration,
                                            "filtered_summary"             : {"total_revenue"                   : total_revenue,
                                                                              "filtered_top20pct_concentration" : filtered_concentration,
                                                                             },
                                            "plots"                        : {"pie_chart"    : pie_chart,
                                                                              "bar_chart"    : bar_chart,
                                                                              "lorenz_curve" : lorenz_curve,
                                                                             }, 
                                           }
        
        return output_dict
        

    def product_kpis(self, filtered_df):
        """
        Calculates Product KPIs grouped by Product and generates visual summaries and returns a dictionary with:
        - product_metrics dataframe
        - summary stats
        - visualizations (bar chart, pie chart, Lorenz curve)
        """
        product_metrics                            = filtered_df.groupby("Product").agg({"Revenue"       : ["sum", "mean", "count"],
                                                                                         "Profit"        : ["sum", "mean"],
                                                                                         "Units Sold"    : ["sum", "mean"],
                                                                                         "Profit_Margin" : "mean",
                                                                                       }).round(2)
    
        product_metrics.columns                    = ["Total_Revenue", 
                                                      "Avg_Revenue", 
                                                      "Transactions",
                                                      "Total_Profit", 
                                                      "Avg_Profit",
                                                      "Total_Units", 
                                                      "Avg_Units",
                                                      "Avg_Profit_Margin",
                                                     ]
    
        product_metrics                            = product_metrics.reset_index()
    
        # Revenue share calculation
        total_revenue                              = product_metrics["Total_Revenue"].sum()
        product_metrics["%Revenue_Share_Filtered"] = (product_metrics["Total_Revenue"] / total_revenue * 100).round(2)
    
        # ARPU and Profitability Metrics
        product_metrics["ARPU"] = (product_metrics["Total_Revenue"] / product_metrics["Total_Units"]).replace([np.inf, -np.inf], 0).round(2)
        product_metrics["Profit_per_Transaction"] = (product_metrics["Total_Profit"] / product_metrics["Transactions"])\
                                                    .replace([np.inf, -np.inf], 0).round(2)
    
        # Sorting for Lorenz curve
        product_metrics                           = product_metrics.sort_values(by        = "Total_Revenue", 
                                                                                ascending = False).reset_index(drop = True)
        
        product_metrics["Cumulative_Revenue_%"]   = (product_metrics["Total_Revenue"].cumsum() / total_revenue * 100).round(2)
    
        # Top 20% concentration
        top_n                                     = max(int(np.ceil(0.2 * len(product_metrics))), 1)
        top_n_revenue                             = product_metrics.loc[:top_n - 1, "Total_Revenue"].sum()
        top20_concentration                       = round((top_n_revenue / total_revenue) * 100, 2)
    
        # Plots
        pie_chart                                 = px.pie(data_frame              = product_metrics,
                                                           names                   = "Product",
                                                           values                  = "%Revenue_Share_Filtered",
                                                           title                   = "% Revenue Share by Product (Filtered)",
                                                           color_discrete_sequence = px.colors.sequential.Plasma,
                                                           width                   = 600,
                                                           height                  = 600,
                                                          )
    
        bar_chart                                 = px.bar(data_frame             = product_metrics,
                                                           x                      = "Product",
                                                           y                      = "Total_Revenue",
                                                           text                   = "%Revenue_Share_Filtered",
                                                           title                  = "Total Revenue by Product (Filtered)",
                                                           color                  = "Total_Revenue",
                                                           color_continuous_scale = "Oranges",
                                                          )
        
        bar_chart.update_traces(texttemplate = '%{text:.2s}%', 
                                textposition = 'outside',
                               )
    
        lorenz_curve                             = px.line(data_frame = product_metrics,
                                                           x          = product_metrics.index + 1,
                                                           y          = "Cumulative_Revenue_%",
                                                           title      = "Cumulative Revenue Distribution (Lorenz Curve - Product)",
                                                           markers    = True,
                                                          )
        
        lorenz_curve.update_layout(xaxis_title = "Product Rank", 
                                   yaxis_title = "Cumulative Revenue %",
                                  )

        output_dict = {"filtered_product_table"          : product_metrics,
                       "filtered_top20pct_concentration" : top20_concentration,
                       "summary"                         : {"total_revenue"          : total_revenue,
                                                            "top20pct_concentration" : top20_concentration,
                                                           },
                       "plots"                           : {"pie_chart"    : pie_chart,
                                                            "bar_chart"    : bar_chart,
                                                            "lorenz_curve" : lorenz_curve,
                                                           },
                      }
    
        return output_dict

    
    def time_series_analysis(self, filtered_df, seasonality_period = 24, min_months_required = 12):
        """
        Perform time series decomposition (Trend, Seasonality, Residual) on monthly revenue,
        with fallback to trend-only analysis if data is insufficient for seasonality and 
        Returns summary table, decomposition components, plots, and key insights
        """
        df                    = filtered_df.copy()
    
        # Ensure Date column is datetime
        df["Date"]            = pd.to_datetime(df["Date"])
        df["Month"]           = df["Date"].dt.to_period("M").dt.to_timestamp()
    
        # Monthly Revenue Aggregation
        monthly               = df.groupby("Month")["Revenue"].sum().reset_index()

        # Remove rows with NaN in Trend for plotting both Revenue & Trend
        if ("Trend" in monthly.columns):
            plot_df = monthly.dropna(subset = ["Trend"])
            y_cols  = ["Revenue", "Trend"]
        
        else:
            plot_df = monthly
            y_cols  = ["Revenue"]

        # Trend plot
        trend_fig            = px.line(data_frame              = plot_df, 
                                       x                       = "Month", 
                                       y                       = y_cols,
                                       title                   = "📈 Monthly Revenue (and Trend if available)",
                                       markers                 = True,
                                      )
        
        trend_fig.update_layout(legend_title_text = "Legend")

        # Initialize optional outputs
        seasonal_fig          = None
        residual_fig          = None
        seasonality_available = False
        avg_seasonal_effect   = None
    
        # Proceed only if enough data
        if (len(monthly) >= 24):
            decomposition       = seasonal_decompose(x                 = monthly["Revenue"], 
                                                     model             = 'additive', 
                                                     period            = 12, 
                                                     extrapolate_trend = 'freq',
                                                    )

            monthly["Trend"]    = decomposition.trend
            monthly["Seasonal"] = decomposition.seasonal
            monthly["Residual"] = decomposition.resid
    
            # Update trend figure to include trend line
            trend_fig             = px.line(data_frame = monthly, 
                                            x          = "Month", 
                                            y          = ["Revenue", "Trend"], 
                                            title      = "📈 Revenue & Trend", 
                                            markers    = True,
                                           )
    
            seasonal_fig          = px.bar(data_frame = monthly, 
                                           x          = "Month", 
                                           y          = "Seasonal", 
                                           title      = "📊 Seasonality",
                                          )
            
            residual_fig          = px.line(data_frame = monthly, 
                                            x          = "Month",
                                            y          = "Residual", 
                                            title      = "🔎 Residual/Noise Component",
                                           )
            
            seasonality_available = True
            avg_seasonal_effect   = monthly["Seasonal"].mean().round(2)
        
        else:
            st.warning("⚠️ Not enough data (<24 months) for seasonality. Showing revenue trend only.")

    
        
    
        summary     = {"avg_monthly_revenue"   : monthly["Revenue"].mean().round(2),
                       "recent_trend_value"    : monthly["Revenue"].iloc[-1].round(2),
                       "seasonality_available" : seasonality_available,
                       "avg_seasonal_effect"   : avg_seasonal_effect
                      }

    
        output_dict = {"time_series_table" : monthly,
                       "plots"             : {"revenue_trend" : trend_fig,
                                              "seasonality"   : seasonal_fig,
                                              "residuals"     : residual_fig,
                                             },
                       "summary"           : summary,
                      }
    
        return output_dict

    
    def covid_impact_analysis(self, filtered_df):
        """
        COVID period-wise performance: revenue, profit, units, efficiency metrics, comparative shares, growth rates, & insightful plots
        """
        covid_summary                           = (filtered_df.groupby("COVID_Period").agg({"Revenue"    : "sum",
                                                                                            "Profit"     : "sum",
                                                                                            "Units Sold" : "sum",
                                                                                            "Date"       : "count"
                                                                                          })\
                                                   .rename(columns = {"Date" : "Transactions"}).round(2).reset_index()
                                                  )
    
        # Efficiency Metrics
        covid_summary["Profit_Margin_%"]        = (covid_summary["Profit"] / covid_summary["Revenue"] * 100).round(2)
        covid_summary["ARPU"]                   = (covid_summary["Revenue"] / covid_summary["Units Sold"]).replace([np.inf, -np.inf], 0).round(2)
        covid_summary["Profit_per_Transaction"] = (covid_summary["Profit"] / covid_summary["Transactions"]).replace([np.inf, -np.inf], 0).round(2)
    
        # Share Metrics
        covid_summary["Revenue_Share_%"]        = (covid_summary["Revenue"] / covid_summary["Revenue"].sum() * 100).round(2)
        covid_summary["Profit_Share_%"]         = (covid_summary["Profit"] / covid_summary["Profit"].sum() * 100).round(2)
        covid_summary["Units_Share_%"]          = (covid_summary["Units Sold"] / covid_summary["Units Sold"].sum() * 100).round(2)
    
        # Optional Growth Rates (vs Pre-COVID baseline)
        baseline                                = covid_summary.loc[covid_summary["COVID_Period"] == "Pre-COVID"]
        
        if not baseline.empty:
            base_revenue                           = baseline["Revenue"].values[0]
            covid_summary["Revenue_vs_PreCOVID_%"] = ((covid_summary["Revenue"] - base_revenue) / base_revenue * 100).round(2)
        
        else:
            covid_summary["Revenue_vs_PreCOVID_%"] = np.nan

        highest_revenue_period                   = covid_summary.loc[covid_summary["Revenue"].idxmax()]["COVID_Period"]
        
        best_profit_margin_period                = covid_summary.loc[covid_summary["Profit_Margin_%"].idxmax()]["COVID_Period"]
        
        # Plots 
        fig_pie         = px.pie(data_frame              = covid_summary,
                                 names                   = "COVID_Period",
                                 values                  = "Revenue_Share_%",
                                 title                   = "🦠 % Revenue Contribution by COVID Period",
                                 color_discrete_sequence = px.colors.sequential.Teal,
                                 width                   = 600,
                                 height                  = 600,
                                )
    
        fig_grouped_bar = px.bar(data_frame              = covid_summary,
                                 x                       = "COVID_Period",
                                 y                       = ["Revenue", "Profit", "Units Sold"],
                                 barmode                 = "group",
                                 title                   = "📊 COVID Period: Revenue, Profit, Units Sold",
                                 text_auto               = True,
                                 color_discrete_sequence = px.colors.qualitative.Set3,
                                )
    
        fig_normalized  = px.bar(data_frame              = covid_summary,
                                 x                       = "COVID_Period",
                                 y                       = ["Revenue_Share_%", "Profit_Share_%", "Units_Share_%"],
                                 barmode                 = "stack",
                                 title                   = "📉 Normalized % Share by COVID Period",
                                 text_auto               = True,
                                 color_discrete_sequence = px.colors.qualitative.Vivid,
                                )
    
        fig_profit_margin = px.line(data_frame = covid_summary,
                                    x          = "COVID_Period",
                                    y          = "Profit_Margin_%",
                                    markers    = True,
                                    title      = "💰 Profit Margin % Trend by COVID Period",
                                   )

        output_dict       = {"covid_summary_table" : covid_summary,
                             "plots"               : {"revenue_pie"         : fig_pie,
                                                      "kpi_grouped_bar"     : fig_grouped_bar,
                                                      "normalized_share"    : fig_normalized,
                                                      "profit_margin_trend" : fig_profit_margin
                                                     },
                             "summary"             : {"highest_revenue_period"    : highest_revenue_period,
                                                      "best_profit_margin_period" : best_profit_margin_period,
                                                     },
                            }
    
        return output_dict


    def customer_segmentation(self, filtered_df, n_clusters: int = 4):
        """
        Industry-grade customer segmentation by country with KPIs, PCA visualizations, and cluster interpretability
        """
        # Aggregation
        segment_df                            = filtered_df.groupby('Country').agg({'Revenue'       : ['sum', 'mean'],
                                                                                    'Units Sold'    : ['sum', 'mean'],
                                                                                    'Profit'        : ['sum', 'mean'],
                                                                                    'Profit_Margin' : 'mean',
                                                                                    'Date'          : 'count',
                                                                                  }).round(2)
    
        segment_df.columns                    = ['Total_Revenue', 
                                                 'Avg_Revenue',
                                                 'Total_Units', 
                                                 'Avg_Units',
                                                 'Total_Profit', 
                                                 'Avg_Profit',
                                                 'Avg_Profit_Margin',
                                                 'Transactions',
                                                ]
        segment_df                            = segment_df.reset_index()
    
        # Derived KPIs
        segment_df["Revenue_per_Transaction"] = (segment_df["Total_Revenue"] / segment_df["Transactions"]).round(2)
        segment_df["Profit_per_Transaction"]  = (segment_df["Total_Profit"] / segment_df["Transactions"]).round(2)
        segment_df["ARPU"]                    = (segment_df["Total_Revenue"] / segment_df["Total_Units"]).replace([np.inf, -np.inf], 0).round(2)
    
        total_revenue                         = segment_df["Total_Revenue"].sum()
        segment_df["Revenue_Share_%"]         = (segment_df["Total_Revenue"] / total_revenue * 100).round(2)
    
        # Feature Selection
        features                              = segment_df[['Total_Revenue', 
                                                            'Avg_Revenue', 
                                                            'Total_Units',
                                                            'ARPU', 
                                                            'Avg_Profit_Margin', 
                                                            'Revenue_per_Transaction', 
                                                            'Profit_per_Transaction',
                                                          ]]
    
        scaler                                = StandardScaler()
        features_scaled                       = scaler.fit_transform(features)
    
        # K-Means Clustering
        kmeans                                = KMeans(n_clusters   = n_clusters, 
                                                       random_state = 42, 
                                                       n_init       = 'auto',
                                                      )
        
        segment_df["Cluster"]                 = kmeans.fit_predict(features_scaled)
    
        # PCA for Dimensionality Reduction (2D Scatter Plot)
        pca                                   = PCA(n_components = 2, 
                                                    random_state = 42,
                                                   )
        components                            = pca.fit_transform(features_scaled)
        segment_df["PCA1"]                    = components[:, 0]
        segment_df["PCA2"]                    = components[:, 1]
    
        # Cluster-wise Aggregation Summary
        cluster_profiles                      = segment_df.groupby("Cluster").agg({'Total_Revenue'     : 'mean',
                                                                                   'Total_Units'       : 'mean',
                                                                                   'Avg_Profit_Margin' : 'mean',
                                                                                   'Revenue_Share_%'   : 'mean',
                                                                                 }).round(2).reset_index()
    
        # PCA Scatter Plot
        fig_scatter = px.scatter(data_frame = segment_df,
                                 x          = "PCA1",
                                 y          = "PCA2",
                                 color      = "Cluster",
                                 hover_data = ["Country", "Total_Revenue", "Total_Units", "Revenue_Share_%"],
                                 title      = "📌 Segmented Countries - PCA Projection",
                                )
    
        # Cluster Size Distribution
        cluster_counts         = segment_df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]

        fig_cluster_dist       = px.bar(data_frame = cluster_counts,
                                        x          = "Cluster", 
                                        y          = "Count",
                                        title      = "👥 Cluster Distribution Count",
                                        text_auto  = True,
                                       )
    
        # Revenue by Cluster Bar
        revenue_by_cluster  = segment_df.groupby("Cluster")["Total_Revenue"].sum().reset_index().sort_values("Total_Revenue", ascending=False)
        fig_revenue_cluster = px.bar(data_frame = revenue_by_cluster,
                                     x          = "Cluster", 
                                     y          = "Total_Revenue",
                                     title      = "💵 Total Revenue by Cluster",
                                     text_auto  = True,
                                    )

        output_dict         = {"segmentation_table" : segment_df,
                               "cluster_profiles"   : cluster_profiles,
                               "plots"              : {"pca_scatter"          : fig_scatter,
                                                       "cluster_distribution" : fig_cluster_dist,
                                                       "revenue_by_cluster"   : fig_revenue_cluster,
                                                      },
                               "summary"            : {"total_clusters"          : n_clusters,
                                                       "largest_cluster_size"    : cluster_counts["Count"].max(),
                                                       "smallest_cluster_size"   : cluster_counts["Count"].min(),
                                                       "highest_revenue_cluster" : revenue_by_cluster.iloc[0]["Cluster"],
                                                      },
                              }
    
        return output_dict

    
    def country_profitability_analysis(self, filtered_df):
        """
        Country profitability analysis with volume context and risk metrics
        """
        country_analysis                         = filtered_df.groupby("Country").agg({"Profit_Margin" : ["mean", "std"],
                                                                                       "Revenue"       : ["sum", "count"],
                                                                                       "Units Sold"    : "sum",
                                                                                       "Date"          : ["min", "max"],
                                                                                     }).reset_index()
        
        # Flatten column names
        country_analysis.columns                 = ["Country", 
                                                    "Avg_Margin", 
                                                    "Margin_Std", 
                                                    "Total_Revenue", 
                                                    "Transactions",
                                                    "Total_Units", 
                                                    "First_Date", 
                                                    "Last_Date",
                                                   ]
        
        # Your existing stability calculation
        country_analysis["Stability"]            = 1 / country_analysis["Margin_Std"]
        
        # Enhanced metrics
        country_analysis["Revenue_Share"]        = (country_analysis["Total_Revenue"] / country_analysis["Total_Revenue"].sum())
        
        country_analysis["Risk_Adjusted_Return"] = (country_analysis["Avg_Margin"] / country_analysis["Margin_Std"]).replace([np.inf, -np.inf], 0)
        
        country_analysis["Avg_Order_Value"]      = (country_analysis["Total_Revenue"] / country_analysis["Transactions"])
        
        country_analysis["Revenue_per_Unit"]     = (country_analysis["Total_Revenue"] / country_analysis["Total_Units"])
        
        # Market presence duration (in days)
        country_analysis["Market_Days"]          = (pd.to_datetime(country_analysis["Last_Date"]) - pd.to_datetime(country_analysis["First_Date"])).dt.days
        
        return country_analysis


    def product_performance_quadrant(self, filtered_df):
        """
        Product performance analysis with growth and volume context
        """
        # Calculate time-based metrics for growth analysis
        filtered_df["Date_dt"]                           = pd.to_datetime(filtered_df["Date"])
        
        # Split data into recent vs older periods for growth calculation
        mid_date                                         = filtered_df["Date_dt"].quantile(0.5)
        recent_data                                      = filtered_df[filtered_df["Date_dt"] >= mid_date]
        older_data                                       = filtered_df[filtered_df["Date_dt"] < mid_date]
        
        # Base product performance (your existing calculation)
        product_performance                              = filtered_df.groupby("Product").agg({"Revenue"       : ["sum", "count"],
                                                                                               "Profit_Margin" : "mean",
                                                                                               "Units Sold"    : "sum"
                                                                                             }).reset_index()
        
        # Flatten columns
        product_performance.columns                      = ["Product", 
                                                            "Total_Revenue", 
                                                            "Transactions", 
                                                            "Avg_Margin", 
                                                            "Total_Units",
                                                           ]
        
        # Your existing metrics
        product_performance["Revenue_Share"]             = (product_performance["Total_Revenue"] / product_performance["Total_Revenue"].sum())
        
        # Enhanced metrics
        product_performance["Transaction_Frequency"]     = product_performance["Transactions"]
        product_performance["Avg_Units_per_Transaction"] = (product_performance["Total_Units"] / product_performance["Transactions"])
        product_performance["Revenue_per_Unit"]          = (product_performance["Total_Revenue"] / product_performance["Total_Units"])
        
        # Growth rate calculation (if sufficient data)
        if ((len(recent_data) > 0)and (len(older_data) > 0)):
            recent_revenue                       = recent_data.groupby("Product")["Revenue"].sum()
            older_revenue                        = older_data.groupby("Product")["Revenue"].sum()
            
            growth_rates                         = ((recent_revenue - older_revenue) / older_revenue * 100).fillna(0)
            product_performance["Growth_Rate_%"] = product_performance["Product"].map(growth_rates).fillna(0)
        
        else:
            product_performance["Growth_Rate_%"] = 0
        
        return product_performance

