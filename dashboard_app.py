# DEPENDENCIES
import warnings
import pandas as pd
import xgboost as xgb
import streamlit as st
import plotly.express as px
from src.data_loader import DataLoader
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_quality_checker import DataQuality
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.exploratory_data_analyzer import SalesEDA
from src.risk_analyzer import RiskSummaryEvaluator
from src.statistical_tester import StatisticalTester
from sklearn.ensemble import GradientBoostingRegressor
from src.time_series_analyzer import TimeSeriesForecaster
from src.machine_learning_modeler import MLModelEvaluator
from src.business_risk_evaluator import BusinessRiskEvaluator
from src.executive_summary_helper import ExecutiveSummaryGenerator
from src.market_segmentation_summarizer import MarketSegmentationSummary


# SUPRESS WARNINGS FOR CLEANER OUTPUT
warnings.filterwarnings('ignore')


# STREAMLIT DASHBOARD
#--------------------------------- CONFIGURATION ---------------------------------
st.set_page_config(page_title            = "Sales Analytics Dashboard",
                   page_icon             = "📊",
                   layout                = "wide",
                   initial_sidebar_state = "expanded",
                   menu_items            = {'Get Help'     : 'https://docs.streamlit.io/',
                                            'Report a bug' : "https://github.com/streamlit/streamlit/issues",
                                            'About'        : "# Professional Sales Analytics Dashboard v2.0",
                                           },
                  )

st.title(body = "\U0001F4CA Sales Performance Dashboard")
st.markdown(body = """This dashboard provides a comprehensive overview of sales performance, market dynamics, and profitability metrics for decision-makers""")


# --------------------------------- CUSTOM CSS ---------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .critical-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html = True)


# --------------------------------- APP START ---------------------------------
uploaded_file = st.file_uploader(label = "Upload Sales Data (.csv or .xlsx)", 
                                 type  = ["csv", "xlsx"],
                                )

if uploaded_file:
    try:
        # Data Loading and Previewing
        loader       = DataLoader(uploaded_file = uploaded_file)
        sales_data   = loader.load_data()
        sales_data   = loader.preprocess_data()
        
        st.success("✅ Sales data loaded and preprocessed successfully!")
        st.subheader("📊 Data Preview")
        
        st.dataframe(data                = sales_data.head(10),
                     use_container_width = True,
                    )
    
        # Data Quality Checking
        data_quality      = DataQuality(dataframe=sales_data)
        basic_report      = data_quality.generate_report()
        anomaly_report    = data_quality.detect_business_logic_anomalies()
        data_completeness = data_quality.validate_data_completeness()
        data_profiling    = data_quality.generate_data_profiling_summary()

        with st.expander(label = "📋 Data Quality Report", expanded = True):
            
            # Basic Metrics Section
            st.subheader(body = "📊 Dataset Overview")
            col1, col2, col3, col4, col5, col6 = st.columns(spec = 6)
            
            rows, cols                         = basic_report['shape']
            
            col1.metric(label = "Dataset Rows",
                        value = f"{rows:,}",
                       )
            
            col2.metric(label = "Dataset Columns", 
                        value = f"{cols}",
                       )
            
            col3.metric(label = "⚠️ Missing Values (Total)",
                        value = basic_report['missing_total'],
                       )
            
            col4.metric(label = "📌 Duplicate Rows",
                        value = basic_report['duplicates'],
                       )
            
            col5.metric(label = "Unique Products",
                        value = basic_report['unique_products'] if basic_report['unique_products'] else "N/A",
                       )
            
            col6.metric(label = "Unique Countries", 
                        value = basic_report['unique_countries'] if basic_report['unique_countries'] else "N/A",
                       )
            
            st.markdown("---")
            
            # Data Quality Score
            st.subheader(body = "🎯 Overall Data Quality Score")
            quality_score                      = data_profiling['data_quality_score']
            
            # Create columns for score display
            score_col1, score_col2, score_col3 = st.columns(spec = [2, 1, 2])
            
            with score_col2:
                if (quality_score >= 90):
                    st.success(f"🟢 **{quality_score}**/100")
                    st.write("Excellent Quality")
                
                elif (quality_score >= 75):
                    st.warning(f"🟡 **{quality_score}**/100") 
                    st.write("Good Quality")
                
                elif (quality_score >= 60):
                    st.warning(f"🟠 **{quality_score}**/100")
                    st.write("Fair Quality")
                
                else:
                    st.error(f"🔴 **{quality_score}**/100")
                    st.write("Poor Quality")
            
            st.markdown("---")
            
            # Column Summary
            st.subheader(body = "📋 Data Type, Missing Values & Outlier Summary")
            st.dataframe(data                = basic_report['column_summary'].style.background_gradient(cmap='Greens'),
                         use_container_width = True,
                        )
            
            st.markdown("---")
            
            # Summary Statistics
            st.subheader(body = "📊 Summary Statistics (Numerical Columns Only)")
            if not basic_report['summary_stats'].empty:
                st.dataframe(data                = basic_report['summary_stats'].style.background_gradient(cmap='Blues'),
                             use_container_width = True,
                            )
            
            else:
                st.info("No numerical columns found in the dataset")
            
            st.markdown("---")
            
            # Business Logic Anomalies
            st.subheader(body = "🔍 Business Logic Anomalies")
            
            # Display summary flags
            for flag in anomaly_report['summary_flags']:
                if ('✅' in flag):
                    st.success(flag)

                elif ('⚠️' in flag):
                    st.warning(flag)
                
                elif ('❗' in flag):
                    st.error(flag)

                else:
                    st.info(flag)
            
            # Detailed anomaly information
            anomaly_col1, anomaly_col2 = st.columns(spec = 2)
            
            with anomaly_col1:
                if (anomaly_report['negative_values']):
                    st.write("**Negative Values Found:**")
                    
                    for col, info in anomaly_report['negative_values'].items():
                        st.write(f"- {col}: {info['count']} rows ({info['percentage']}%)")
                
                if (anomaly_report['zero_values']):
                    st.write("**Zero Values in Critical Columns:**")
                    for col, info in anomaly_report['zero_values'].items():
                        st.write(f"- {col}: {info['count']} rows ({info['percentage']}%)")
            
            with anomaly_col2:
                if (anomaly_report['inconsistent_calculations']):
                    st.write("**Calculation Inconsistencies:**")
                    for calc_type, info in anomaly_report['inconsistent_calculations'].items():

                        if (calc_type == 'profit_calculation'):
                            st.write(f"- Profit calculation issues: {info['discrepancy_count']} rows")
                        
                        elif (calc_type == 'unit_price_variation'):
                            st.write(f"- Unit price variation: {info['coefficient_variation']}% CV")
                
                if (anomaly_report['extreme_ratios']):
                    st.write("**Extreme Ratios Found**")
                    
                    for ratio_type, info in anomaly_report['extreme_ratios'].items():
                        if (ratio_type == 'profit_margins'):
                            if (info['negative_margins'] > 0):
                                st.write(f"- Negative margins: {info['negative_margins']} rows")
                            
                            if (info['super_high_margins'] > 0):
                                st.write(f"- Super high margins (>95%): {info['super_high_margins']} rows")
            
            st.markdown("---")
            
            # Data Completeness Analysis
            st.subheader(body = "📈 Data Completeness Analysis")
            
            completeness_score              = data_completeness['completeness_score']
            comp_col1, comp_col2, comp_col3 = st.columns(spec = 3)
            
            with comp_col1:
                st.metric(label = "Completeness Score", 
                          value = f"{completeness_score}%",
                         )
            
            with comp_col2:
                presence_rate = data_completeness['column_presence']['presence_rate']
                st.metric(label = "Column Presence", 
                          value = f"{presence_rate}%",
                         )
            
            with comp_col3:
                if data_completeness['date_continuity']:
                    continuity = data_completeness['date_continuity']['continuity_percentage']
                    st.metric(label = "Date Continuity", 
                              value = f"{continuity}%",
                             )

                else:
                    st.metric(label = "Date Continuity", 
                              value = "N/A",
                             )
            
            # Missing columns
            if (data_completeness['column_presence']['missing_columns']):
                st.error(f"Missing Required Columns: {', '.join(data_completeness['column_presence']['missing_columns'])}")
            
            # Missing data by column
            st.write("**Missing Data by Column:**")
            missing_data_df = pd.DataFrame([{'Column'        : col,
                                             'Missing Count' : info['missing_count'],
                                             'Missing %'     : info['missing_percentage'],
                                             'Complete %'    : info['completeness_percentage']
                                            } for col, info in data_completeness['missing_data_analysis'].items()
                                          ])
            
            if (not missing_data_df.empty):
                st.dataframe(missing_data_df.style.background_gradient(subset = ['Complete %'], cmap = 'RdYlGn'),
                             use_container_width = True)
            
            # Recommendations
            st.write("**Recommendations:**")
            for rec in data_completeness['recommendations']:
                if ('✅' in rec):
                    st.success(rec)
                
                elif ('⚠️' in rec):
                    st.warning(rec)
                
                elif ('❗' in rec):
                    st.error(rec)

                else:
                    st.info(rec)
            
            st.markdown("---")
            
            # Dataset Profiling Summary
            st.subheader(body = "📋 Dataset Profiling Summary")
            
            overview                   = data_profiling['dataset_overview']
            
            profile_col1, profile_col2 = st.columns(spec = 2)
            
            with profile_col1:
                st.write("**Dataset Overview:**")
                st.write(f"- Total Records: {overview['total_records']:,}")
                st.write(f"- Total Columns: {overview['total_columns']}")
                st.write(f"- Memory Usage: {overview['memory_usage_mb']} MB")
                
                if overview['date_range']:
                    st.write(f"- Date Range: {overview['date_range']['start_date']} to {overview['date_range']['end_date']}")
                    st.write(f"- Days Covered: {overview['date_range']['days_covered']}")
            
            with profile_col2:
                if overview['primary_metrics']:
                    st.write("**Primary Business Metrics:**")
                    for metric, values in overview['primary_metrics'].items():
                        st.write(f"**{metric}:**")
                        st.write(f"  - Total: {values['total']:,}")
                        st.write(f"  - Average: {values['average']:,}")
                        st.write(f"  - Range: {values['min']:,} to {values['max']:,}")
            
            # Categorical Analysis
            if data_profiling['categorical_analysis']:
                st.write("**Categorical Columns Analysis:**")
                cat_analysis_df = pd.DataFrame([{'Column'                : col,
                                                 'Unique Values'         : info['unique_values'],
                                                 'Most Frequent'         : info['most_frequent'],
                                                 'Most Frequent Count'   : info['most_frequent_count'],
                                                 'Distribution Evenness' : info['distribution_evenness']
                                                } for col, info in data_profiling['categorical_analysis'].items()
                                              ])
                st.dataframe(data                = cat_analysis_df, 
                             use_container_width = True,
                            )
            
            # Numerical Analysis
            if (data_profiling['numerical_analysis']):
                st.write("**Numerical Columns Analysis:**")
                num_analysis_df = pd.DataFrame([{'Column'    : col,
                                                 'Mean'      : info['mean'],
                                                 'Median'    : info['median'],
                                                 'Std Dev'   : info['std_dev'],
                                                 'CV %'      : info['coefficient_variation'],
                                                 'Skewness'  : info['skewness'],
                                                 'Outlier %' : info['outlier_percentage'],
                                                } for col, info in data_profiling['numerical_analysis'].items()
                                              ])
                st.dataframe(data                = num_analysis_df.style.background_gradient(cmap='viridis'),
                             use_container_width = True,
                            )
            
            # Relationships Analysis
            if (data_profiling['relationships_analysis']):
                st.write("**Key Relationships:**")
                for relationship, correlation in data_profiling['relationships_analysis'].items():
                    correlation_strength = ""
                    
                    if (abs(correlation) >= 0.8):
                        correlation_strength = "Very Strong"
                    
                    elif (abs(correlation) >= 0.6):
                        correlation_strength = "Strong"
                    
                    elif (abs(correlation) >= 0.4):
                        correlation_strength = "Moderate"

                    elif (abs(correlation) >= 0.2):
                        correlation_strength = "Weak"

                    else:
                        correlation_strength = "Very Weak"
                    
                    st.write(f"- {relationship.replace('_', ' ').title()}: {correlation:.3f} ({correlation_strength})")


        # EXPLORATORY DATA ANALYSIS
        with st.expander(label = "🔍 Exploratory Data Analysis Report", expanded = True):
            st.markdown("---")
            st.header("📊 Interactive Sales Analytics")
    
            # Initialize SalesEDA after preprocessing
            sales_eda                    = SalesEDA(sales_data)
            filtered_data                = sales_eda.data_filtering()
    
            # Streamlit Tabs for Dashboard Segmentation
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs = ["🌍 Market KPIs", 
                                                                 "🛍️ Product KPIs", 
                                                                 "📈 Time Series Analysis",
                                                                 "🦠 COVID Impact", 
                                                                 "👥 Customer Segmentation",
                                                                 "📊 Profitability & Quadrants"
                                                                ]
                                                        )
    
            with tab1:
                st.subheader(body = "🌍 Market KPIs (Filtered View)")
                
                market_kpis = sales_eda.market_kpis(filtered_data)
                st.dataframe(data                = market_kpis['filtered_market_table'], 
                             use_container_width = True,
                            )
                
                st.plotly_chart(figure_or_data      = market_kpis['plots']['pie_chart'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = market_kpis['plots']['bar_chart'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = market_kpis['plots']['lorenz_curve'], 
                                use_container_width = True,
                               )
    
            with tab2:
                st.subheader(body = "🛍️ Product KPIs (Filtered View)")
                
                product_kpis = sales_eda.product_kpis(filtered_data)
                
                st.dataframe(data                = product_kpis['filtered_product_table'], 
                             use_container_width = True,
                            )
                
                st.plotly_chart(figure_or_data      = product_kpis['plots']['pie_chart'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = product_kpis['plots']['bar_chart'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = product_kpis['plots']['lorenz_curve'], 
                                use_container_width = True,
                               )
    
            with tab3:
                st.subheader(body = "📈 Time Series Revenue Decomposition")
                
                time_series_kpis   = sales_eda.time_series_analysis(filtered_data)
                
                monthly_table      = time_series_kpis["time_series_table"]
                summary            = time_series_kpis["summary"]
                plots              = time_series_kpis["plots"]
    
                st.markdown(body = "#### 🗓️ Monthly Aggregated Revenue")
                
                st.dataframe(data                = monthly_table, 
                             use_container_width = True,
                            )
    
                # Display Key Insights
                st.markdown(f"""
                #### 📌 Key Insights:
                - **Average Monthly Revenue:** ${summary["avg_monthly_revenue"]:,}
                - **Most Recent Revenue Value:** ${summary["recent_trend_value"]:,}
                """)
            
                if summary["seasonality_available"]:
                    st.markdown(f"- **Average Seasonal Effect:** ${summary['avg_seasonal_effect']:,}")
                
                else:
                    st.warning("⚠️ Not enough historical data (minimum 24 months) for reliable seasonality analysis")
    
                # Trend Plot (Always Shown)
                st.plotly_chart(figure_or_data      = plots["revenue_trend"], 
                                use_container_width = True,
                               )
    
                # Optional Seasonal/Residual Plots
                if summary["seasonality_available"]:
                    st.plotly_chart(figure_or_data      = plots["seasonality"], 
                                    use_container_width = True,
                                   )
                    
                    st.plotly_chart(figure_or_data      = plots["residuals"], 
                                    use_container_width = True,
                                   )
    
                    st.plotly_chart(figure_or_data      = plots["revenue_trend"], 
                                    use_container_width = True,
                                   )
    
            with tab4:
                st.subheader(body = "🦠 COVID Impact Analysis (Filtered View)")
                
                covid_kpis = sales_eda.covid_impact_analysis(filtered_data)
                
                st.dataframe(data                = covid_kpis['covid_summary_table'], 
                             use_container_width = True,
                            )
                
                st.plotly_chart(figure_or_data      = covid_kpis['plots']['revenue_pie'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = covid_kpis['plots']['kpi_grouped_bar'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = covid_kpis['plots']['normalized_share'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = covid_kpis['plots']['profit_margin_trend'], 
                                use_container_width = True,
                               )
    
            with tab5:
                st.subheader(body = "👥 Customer Segmentation (Filtered View)")
                
                customer_segmentation = sales_eda.customer_segmentation(filtered_data, 
                                                                        n_clusters = len(filtered_data['Country'].unique()),
                                                                       )
                
                st.dataframe(data                = customer_segmentation['segmentation_table'], 
                             use_container_width = True,
                            )
                
                st.plotly_chart(figure_or_data      = customer_segmentation['plots']['pca_scatter'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = customer_segmentation['plots']['cluster_distribution'], 
                                use_container_width = True,
                               )
                
                st.plotly_chart(figure_or_data      = customer_segmentation['plots']['revenue_by_cluster'], 
                                use_container_width = True,
                               )


            with tab6:
                st.subheader(body = "Time-based Country Performance")

                monthly_country          = filtered_data.groupby([pd.to_datetime(filtered_data["Date"]).dt.to_period("M"), "Country"])["Revenue"].sum().reset_index()
                monthly_country["Month"] = monthly_country["Date"].dt.to_timestamp()
    
                country_trends           = px.line(data_frame = monthly_country,
                                                   x          = "Month",
                                                   y          = "Revenue",
                                                   color      = "Country",
                                                   title      = "Monthly Revenue Trends by Country",
                                                   markers    = True,
                                                  )

                st.plotly_chart(figure_or_data      = country_trends,
                                use_container_width = True,
                               )

                st.subheader(body = "🌍 Country Profitability Analysis")
                country_data    = sales_eda.country_profitability_analysis(filtered_df = filtered_data)
                
                country_treemap = px.treemap(data_frame             = country_data,
                                             path                   = ["Country"],
                                             values                 = "Total_Revenue",
                                             color                  = "Risk_Adjusted_Return",
                                             color_continuous_scale = "RdYlBu",
                                             title                  = "Country Profitability: Size=Revenue, Color=Risk-Adjusted Return",
                                             hover_data             = ["Avg_Margin", "Stability", "Avg_Order_Value"],
                                            )
                
                st.plotly_chart(figure_or_data      = country_treemap,
                                use_container_width = True,
                               )

                st.subheader(body = "Country Risk-Return Matrix")
                risk_return = px.scatter(data_frame = country_data,
                                         x          = "Margin_Std",
                                         y          = "Avg_Margin",
                                         size       = "Total_Revenue",
                                         color      = "Revenue_Share",
                                         hover_data = ["Country", "Avg_Order_Value", "Market_Days"],
                                         title      = "Country Risk-Return: X=Volatility, Y=Avg Margin, Size=Revenue",
                                         labels     = {"Margin_Std": "Risk (Margin Volatility)", "Avg_Margin": "Return (Avg Margin)"},
                                        )

                st.plotly_chart(figure_or_data      = risk_return,
                                use_container_width = True,
                               )

                st.subheader(body = "📊 Product Performance Quadrant")
                product_data     = sales_eda.product_performance_quadrant(filtered_df = filtered_data)

                product_quadrant = px.scatter(data_frame             = product_data,
                                              x                      = "Revenue_Share",
                                              y                      = "Avg_Margin",
                                              size                   = "Transaction_Frequency",
                                              color                  = "Growth_Rate_%",
                                              hover_data             = ["Product", "Revenue_per_Unit", "Avg_Units_per_Transaction"],
                                              title                  = "Product Performance: X=Revenue Share, Y=Margin, Size=Frequency, Color=Growth%",
                                              color_continuous_scale = "RdYlGn",
                                             )
    
                # Add quadrant lines
                product_quadrant.add_hline(y          = product_data["Avg_Margin"].median(), 
                                           line_dash  = "dash", 
                                           line_color = "gray",
                                          )

                product_quadrant.add_vline(x          = product_data["Revenue_Share"].median(), 
                                           line_dash  = "dash", 
                                           line_color = "gray",
                                          )

                st.plotly_chart(figure_or_data      = product_quadrant, 
                                use_container_width = True,
                               )
            
        # Market Summary
        market_summary = MarketSegmentationSummary(dataframe = sales_data)
        market_metrics = market_summary.calculate_market_metrics()
            
        with st.expander(label = "📊 Market Segmentation Results", expanded = True):
            st.markdown(body = "#### 📊 Market Summary Table")
                
            st.dataframe(data                = market_metrics["market_table"].set_index('Country').T.style.background_gradient(cmap = "Greens"),
                         use_container_width = True,
                        )
            
            # Horizontal Bar Plot for Revenue
            st.markdown(body = "#### 💰 Total Revenue by Country")
            
            fig_rev = px.bar(data_frame             = market_metrics["country_wise_revenue"].sort_values("Total_Revenue"),
                             x                      = "Total_Revenue", 
                             y                      = "Country", 
                             orientation            = "h", 
                             color                  = "Total_Revenue",
                             color_continuous_scale = "Blues",
                            )
                
            st.plotly_chart(figure_or_data      = fig_rev, 
                            use_container_width = True,
                           )
            
            # Pie Chart for % Revenue Share
            st.markdown(body = "#### 🥧 % Revenue Contribution")
            fig_pie = px.pie(data_frame              = market_metrics["pie_data"], 
                             names                   = "Country", 
                             values                  = "%Revenue_Contribution",
                             color_discrete_sequence = px.colors.sequential.RdBu,
                             width                   = 600,
                             height                  = 600,
                            )
                
            st.plotly_chart(figure_or_data      = fig_pie, 
                            use_container_width = True,
                           )
            
            # Top 3 markets
            st.markdown(body = "#### 🏆 Top 3 Countries by Total Revenue")
            
            top_3_countries = market_metrics["top3_by_revenue"].set_index('Country').T
            
            st.dataframe(data                = top_3_countries.style.background_gradient(cmap = "Purples"),
                         use_container_width = True,
                        )
            
            # Affinity Matrix
            affinity_matrix = market_summary.generate_product_country_affinity_matrix()
            st.markdown(body = "#### 🔗 Product-Country Revenue Affinity Matrix")
            
            st.dataframe(data                = affinity_matrix.style.background_gradient(cmap = "YlGnBu"), 
                         use_container_width = True,
                        )

            # Country Value Segmentation
            value_segment_df = market_summary.country_value_segmentation()
            st.markdown(body = "#### 🏷️ Country Value Segmentation (High / Mid / Low)")
            
            fig_seg = px.bar(data_frame         = value_segment_df.sort_values("Revenue"),
                             x                  = "Revenue",
                             y                  = "Country",
                             color              = "Value_Segment",
                             orientation        = "h",
                             color_discrete_map = {"High": "green", "Mid": "orange", "Low": "red"},
                            )

            st.plotly_chart(figure_or_data      = fig_seg, 
                            use_container_width = True,
                           )
            
        with st.expander(label = "⏳ Time Series & Forecasting", expanded = True):
            st.subheader(body = "📈 Revenue Forecasting Options")
        
            model_options  = ["ARIMA", 
                              "Exponential Smoothing", 
                              "Holt Linear Trend", 
                              "Simple Average Forecast", 
                              "Naive Forecast",
                             ]
            
            model_choice   = st.selectbox("Select Forecasting Model:", model_options)
        
            forecast_steps = st.slider(label     = "Select Forecast Horizon (Months):", 
                                       min_value = 3, 
                                       max_value = 12, 
                                       value     = 6,
                                      )
        
            forecaster     = TimeSeriesForecaster(filtered_data)
        
            if (model_choice == "ARIMA"):
                p        = st.number_input("ARIMA p:", 0, 5, 1)
                d        = st.number_input("ARIMA d:", 0, 2, 1)
                q        = st.number_input("ARIMA q:", 0, 5, 1)
                forecast = forecaster.arima_forecast(steps = forecast_steps,
                                                     order = (p,d,q),
                                                    )
            
            elif (model_choice == "Exponential Smoothing"):
                forecast = forecaster.exponential_smoothing_forecast(steps = forecast_steps)
                
            elif (model_choice == "Holt Linear Trend"):
                forecast = forecaster.holt_linear_forecast(steps = forecast_steps)
            
            elif (model_choice == "Simple Average Forecast"):
                forecast = forecaster.simple_avg_forecast(steps = forecast_steps)
                 
            elif (model_choice == "Naive Forecast"):
                forecast = forecaster.naive_forecast(steps = forecast_steps)
            
            else:
                st.warning("Invalid Model Selected")
                forecast = None
        
            if forecast is not None:
                forecast_plot = forecaster.generate_plot(forecast, 
                                                         model_name = model_choice, 
                                                         steps      = forecast_steps,
                                                        )
                
                st.plotly_chart(figure_or_data      = forecast_plot,
                                use_container_width = True,
                               )
        
                st.subheader(body = "📊 Forecast Data")
                forecast_df = pd.DataFrame({"Date"               : forecast.index,
                                            "Forecasted Revenue" : forecast.values.round(2),
                                          })
                
                st.dataframe(data                = forecast_df, 
                             use_container_width = True,
                            )

            st.markdown("---")
            st.subheader("📅 Seasonal Volume Analysis")
            target_col   = st.selectbox(label   = "Select Metric for Seasonal Analysis:", 
                                        options = ["Units Sold", "Revenue", "Cost", "Profit", "Profit_Margin"],
                                       )

            seasonal_fig = forecaster.seasonal_volume_forecast(target_column = target_col)

            st.plotly_chart(figure_or_data      = seasonal_fig, 
                            use_container_width = True,
                           )

        with st.expander("📈 Statistical Testing (Normality, ANOVA, Group Differences)", expanded = True):
            st.subheader(body = "📊 Statistical Test Summary")
        
            # User Inputs
            metric_to_test = st.selectbox("Select Metric:", ['Revenue', 'Profit', 'Units Sold', 'Profit_Margin'], index = 0)
            group_by_var   = st.selectbox("Group By:", ['Country', 'Product'], index=0)
        
            st.markdown(body = f"#### 🧪 Testing `{metric_to_test}` grouped by `{group_by_var}`")
        
            stats_tester   = StatisticalTester(filtered_data)
            grouped_data   = stats_tester.get_grouped_data(metric_to_test, group_by_var)
        
            # Shapiro Test 
            shapiro_stat, shapiro_p = stats_tester.shapiro_test(metric_to_test)
            
            st.markdown(body = "**Shapiro-Wilk Test (Normality)**")
            st.info(f"W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")
            
            if (shapiro_p < 0.05):
                st.warning("❗ Data does **not** follow normal distribution (p < 0.05). Non-parametric tests recommended")
            
            else:
                st.success("✅ Data appears **normally distributed** (p ≥ 0.05). Parametric tests applicable.")
        
            # ANOVA Test
            anova_f, anova_p = stats_tester.anova_test(grouped_data)
            
            st.markdown(body = "**ANOVA Test (Mean Differences)**")
            st.info(f"F = {anova_f:.2f}, p = {anova_p:.4f}")
            
            if (anova_p < 0.05):
                st.error("❗ **Significant differences exist** between groups (p < 0.05)")
            
            else:
                st.success("✅ No significant differences detected (p ≥ 0.05)")
        
            # Levene's Test
            levene_stat, levene_p = stats_tester.levene_test(grouped_data)
            
            st.markdown(body = "**Levene's Test (Variance Equality)**")
            st.info(f"Statistic = {levene_stat:.2f}, p = {levene_p:.4f}")
            
            if (levene_p < 0.05):
                st.warning("⚠️ **Unequal variances detected** (p < 0.05). Consider robust testing methods")
            
            else:
                st.success("✅ Variances appear equal (p ≥ 0.05)")
        
            # Optional Non-Parametric ---
            if (shapiro_p < 0.05):
                st.markdown(body = "#### 🧮 Non-Parametric Test Recommendation (Kruskal-Wallis):")
                
                kruskal_h, kruskal_p = stats_tester.kruskal_test(grouped_data)
                
                st.info(f"H = {kruskal_h:.2f}, p = {kruskal_p:.4f}")
                
                if (kruskal_p < 0.05):
                    st.error("❗ **Significant group differences** detected by Kruskal-Wallis test")
                
                else:
                    st.success("✅ Kruskal-Wallis test shows **no significant differences**")
        
            # Optional Mann-Whitney Test (only for 2 groups)
            if (len(grouped_data) == 2):
                st.markdown(body = "#### 🔎 Mann-Whitney U-Test (2 groups only):")
                
                u_stat, u_p = stats_tester.mann_whitney_test(grouped_data[0], grouped_data[1])
                st.info(f"U = {u_stat:.2f}, p = {u_p:.4f}")
                
                if (u_p < 0.05):
                    st.error("❗ **Significant difference detected** (p < 0.05)")
                
                else:
                    st.success("✅ **No significant difference detected** (p ≥ 0.05)")
        
            # Boxplot
            st.markdown(body = "#### 📊 Group-wise Boxplot:")
            box_fig = stats_tester.boxplot_figure(metric_to_test, group_by_var)
            
            st.plotly_chart(figure_or_data      = box_fig, 
                            use_container_width = True,
                           )

            st.markdown(body = "### 🔍 Estimate Price Elasticity of Demand")
            numeric_cols                        = filtered_data.select_dtypes(include = ['number']).columns.tolist()

            price_col                           = st.selectbox(label    = "Select Price Column:", 
                                                                options = numeric_cols,
                                                              )

            demand_col                          = st.selectbox(label   = "Select Demand Column:", 
                                                               options = numeric_cols, index=1 if len(numeric_cols) > 1 else 0,
                                                              )
            
            elasticity, summary, elasticity_fig = stats_tester.estimate_price_elasticity(price_col, demand_col)

            st.success(f"**Estimated Price Elasticity**: {elasticity:.2f}")
            st.text(summary.as_text())
                    
            st.plotly_chart(figure_or_data      = elasticity_fig, 
                            use_container_width = True,
                           )

            st.caption("ℹ️ Elasticity measures how responsive quantity demanded is to price changes")

            st.caption("ℹ️ This interactive statistical testing section dynamically adjusts to your selections and distribution properties")


        with st.expander(label = "🤖 Machine Learning Model Comparison", expanded = True):
            
            # Machine Learning Data Preparation
            df_ml                = filtered_data.copy()
            df_ml["Month"]       = df_ml["Date"].dt.month
            le_country           = LabelEncoder().fit(df_ml["Country"])
            le_product           = LabelEncoder().fit(df_ml["Product"])

            df_ml["Country_Enc"] = le_country.transform(df_ml["Country"])
            df_ml["Product_Enc"] = le_product.transform(df_ml["Product"])

            # Target Variable Selection
            possible_targets     = ['Revenue', 'Profit', 'Units Sold', 'Profit_Margin']
            target               = st.selectbox(label   = "🎯 Select Target Variable:", 
                                                options = possible_targets, 
                                                index   = 0)

            st.subheader(body = f"📊 Select Models for Predicting {target}")
        
            # Automatically Define Feature Set
            exclude_columns      = [target, 'Date'] + [col for col in ['Customer_ID'] if col in df_ml.columns]
            feature_list         = [column for column in df_ml.columns if column not in exclude_columns and df_ml[column].dtype in [int, float]]
        
            st.markdown(body = f"✅ **Selected Features ({len(feature_list)}):** {', '.join(sorted(feature_list))}")

            model_options        = {"Linear Regression" : LinearRegression(),
                                    "Lasso Regression"  : Lasso(alpha = 0.1, random_state = 42),
                                    "Decision Tree"     : DecisionTreeRegressor(random_state = 42),
                                    "Random Forest"     : RandomForestRegressor(n_estimators = 100, random_state = 42),
                                    "Gradient Boosting" : GradientBoostingRegressor(n_estimators = 100, random_state = 42),
                                    "XGBoost"           : xgb.XGBRegressor(n_estimators = 100, random_state = 42),
                                   }
    
            selected_models      = st.multiselect(label   = "Choose models to run:", 
                                                  options = list(model_options.keys()), 
                                                  default = ["Random Forest"],
                                                 )
    
            if selected_models:
                evaluator   = MLModelEvaluator(df_ml, target, feature_list, scale_data=True)
                all_results = list()
        
                def prediction_plot(y_true, y_pred, model_name):
                    df_plot = pd.DataFrame({'Actual'    : y_true, 
                                            'Predicted' : y_pred,
                                          })
                    
                    fig     = px.scatter(data_frame               = df_plot, 
                                         x                        ='Actual', 
                                         y                        = 'Predicted', 
                                         title                    = f"📈 Actual vs Predicted: {model_name}",
                                         labels                   = {'Actual': 'Actual', 'Predicted': 'Predicted'},
                                         trendline                = "ols", 
                                         trendline_color_override = "firebrick",
                                        )
                    
                    fig.add_shape(type = 'line', 
                                  x0   = y_true.min(), 
                                  y0   = y_true.min(),
                                  x1   = y_true.max(), 
                                  y1   = y_true.max(),
                                  line = dict(color = 'green', 
                                              dash  = 'dash',
                                             )
                                 )
                    
                    fig.update_layout(showlegend = False)
                    return fig
        
                for model_name in selected_models:
                    model  = model_options[model_name]
                    result = evaluator.evaluate_model(model, 
                                                      model_name,
                                                     )
                    all_results.append(result)
        
                    st.markdown(f"#### 🔎 {model_name}")
                    r2, cv_r2, rmse = result['r2'], result['cv_r2'], result['rmse']
                    mean_target     = evaluator.y_test.mean()
        
                    st.info(f"📌 **R²:** {r2:.3f} | **CV R²:** {cv_r2:.3f} | **RMSE:** {rmse:.2f}")
        
                    # Automated Insights
                    insights = list()
                    if (r2 > 0.85):
                        insights.append("✅ **High explanatory power (R² > 0.85)**")
                    
                    elif (r2 > 0.70):
                        insights.append("⚠️ **Moderate R² (0.70 - 0.85)**")
                    
                    else:
                        insights.append("❗ **Low R² (< 0.70)** — consider feature engineering")
        
                    gap = abs(r2 - cv_r2)
                    
                    if (gap <= 0.05):
                        insights.append("✅ **Good generalization (small CV gap)**")
                    
                    elif (gap > 0.1):
                        insights.append("⚠️ **Potential overfitting (CV gap > 0.1)**")
        
                    if (rmse < 0.1 * mean_target):
                        insights.append("✅ **Low error relative to target mean**")
                    
                    else:
                        insights.append(f"⚠️ **RMSE ({rmse:.2f}) is significant vs target mean ({mean_target:.2f})**")
        
                    for i in insights:
                        st.markdown(i)
        
                    # Feature Importance
                    fig_importance = evaluator.feature_importance_plot(result['model'])
                    
                    if fig_importance:
                        st.plotly_chart(figure_or_data      = fig_importance, 
                                        use_container_width = True,
                                       )

                    # Actual vs Predicted
                    st.plotly_chart(figure_or_data      = prediction_plot(evaluator.y_test, 
                                                                          result['preds'], 
                                                                          model_name,
                                                                         ), 
                                    use_container_width = True,
                                   )
        
                # Summary Table
                st.markdown(body = "### 📊 Summary Comparison Table")
                summary_df = pd.DataFrame([{"Model" : res['name'], 
                                            "R²"    : res['r2'], 
                                            "CV R²" : res['cv_r2'], 
                                            "RMSE"  : res['rmse'],
                                           } for res in all_results])
                
                st.dataframe(data = summary_df)

                # Automated Summary Insights
                st.markdown(body = "### 📝 Automated Summary Insights")
                
                if not summary_df.empty:
                    best_r2_model     = summary_df.loc[summary_df['R²'].idxmax()]
                    lowest_rmse_model = summary_df.loc[summary_df['RMSE'].idxmin()]
                    biggest_gap_model = summary_df.iloc[(summary_df["R²"] - summary_df["CV R²"]).abs().idxmax()]
                
                    st.success(f"✅ **Best R²:** {best_r2_model['Model']} with R² = {best_r2_model['R²']:.3f}")
                
                    st.info(f"✅ **Lowest RMSE:** {lowest_rmse_model['Model']} with RMSE = {lowest_rmse_model['RMSE']:.2f}")
                
                    gap = abs(biggest_gap_model["R²"] - biggest_gap_model["CV R²"])
                    
                    if (gap > 0.1):
                        st.warning(f"⚠️ **Potential Overfitting Detected:** {biggest_gap_model['Model']} with R²-CV R² gap = {gap:.3f}")
                
                    # Recommendation Logic
                    recommendation = summary_df.sort_values(by        = ["R²", "RMSE"], 
                                                            ascending = [False, True],
                                                           ).iloc[0]
                    st.markdown(f"👑 **Recommended Model:** **{recommendation['Model']}**, with R² = {recommendation['R²']:.3f} and RMSE = {recommendation['RMSE']:.2f}")
                
                else:
                    st.info("Summary table is empty. No insights to display")
        
            else:
                st.warning("⚠️ Please select at least one model")


        with st.expander(label = "⚠️ KPIs + Comprehensive Risk Summary", expanded = True):
            st.subheader(body = "📊 Key Business KPIs")
        
            # Initialize Evaluator
            risk_eval        = RiskSummaryEvaluator(filtered_df = filtered_data, 
                                                    total_df    = sales_data,
                                                   )
        
            # KPI Deltas
            col1, col2, col3 = st.columns(3)
        
            col1.metric(label       = "Total Revenue",
                        value       = f"${filtered_data['Revenue'].sum():,.0f}",
                        delta       = f"{((filtered_data['Revenue'].sum() - risk_eval.total_revenue_overall) / risk_eval.total_revenue_overall) * 100:.2f}%",
                        delta_color = "inverse",
                       )
        
            col2.metric(label       = "Total Profit",
                        value       = f"${filtered_data['Profit'].sum():,.0f}",
                        delta       = f"{((filtered_data['Profit'].sum() - risk_eval.total_profit_overall) / risk_eval.total_profit_overall) * 100:.2f}%",
                        delta_color = "inverse",
                       )
        
            col3.metric(label = "Avg Profit Margin",
                        value = f"{filtered_data['Profit_Margin'].mean():.2f}%",
                        delta = f"{filtered_data['Profit_Margin'].mean() - risk_eval.avg_profit_margin_overall:.2f} pp",
                       )
        
            st.markdown("---")
            st.subheader(body = "📉 Revenue Volatility")
            st.info(f"Revenue Coefficient of Variation (CV): {risk_eval.volatility:.2f}%")
            
            st.markdown("---")
            st.subheader(body = "📊 Concentration Risks")
            st.metric(label = "Market Concentration (HHI)",
                      value = f"{risk_eval.hhi_country:.0f}",
                     )
            
            st.metric(label = "Product Concentration (HHI)",
                      value = f"{risk_eval.hhi_product:.0f}",
                     )
            
            st.caption("Interpretation: HHI > 2500 → High Risk, 1500–2500 → Moderate, <1500 → Low")
            
            st.markdown("---")
            st.subheader(body = "📉 Profitability Risk")
            st.metric(label = "Low Margin Transactions (%)",
                      value = f"{risk_eval.low_margin_pct:.1f}%",
                     )
            
            st.caption(f"High values (>{risk_eval.low_margin_threshold}%) indicate low-margin exposure")
            
            st.markdown("---")
            st.subheader(body = "📊 Performance Trend Risk")
            st.metric(label = "Recent vs Historical Revenue Change (%)",
                      value = f"{risk_eval.perf_decline:.1f}%",
                     )
            
            st.markdown("---")
            st.subheader(body = "🔍 Low Margin Transaction Analysis")
            
            # Get low margin analysis
            low_margin_analysis    = risk_eval.detect_low_margin_transactions()
            
            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(spec = 4)
            
            col1.metric(label = "Low Margin Count",
                        value = f"{low_margin_analysis['low_margin_count']:,}",
                        help  = f"Out of {low_margin_analysis['total_transactions']:,} total transactions",
                       )
            
            col2.metric(label = "Revenue Impact",
                        value = f"{low_margin_analysis['revenue_impact_pct']:.1f}%",
                        help  = "Percentage of total revenue from low-margin transactions",
                       )
            
            col3.metric(label = "Profit Impact", 
                        value = f"{low_margin_analysis['profit_impact_pct']:.1f}%",
                        help  = "Percentage of total profit from low-margin transactions",
                       )
            
            col4.metric(label = "Avg Low Margin",
                        value = f"{low_margin_analysis['avg_low_margin']:.1f}%",
                        help  = f"Average margin of transactions below {low_margin_analysis['threshold']}%",
                       )
            
            # Display breakdown tables
            if (not low_margin_analysis['country_breakdown'].empty):
                st.write("**Top-5 Countries by Low-Margin Revenue:**")
                country_top = low_margin_analysis['country_breakdown'].nlargest(5, 'Revenue')
                st.dataframe(country_top, 
                             use_container_width = True)
            
            if (not low_margin_analysis['product_breakdown'].empty):
                st.write("**Top-5 Products by Low-Margin Revenue:**")
                product_top = low_margin_analysis['product_breakdown'].nlargest(5, 'Revenue')
                st.dataframe(product_top, 
                             use_container_width = True)
            
            st.markdown("---")
            st.subheader(body = "📈 Return Risk Analysis")
            
            # Get return risk analysis
            return_risk            = risk_eval.calculate_return_risk()
            
            # Display return risk metrics
            col1, col2, col3, col4 = st.columns(spec = 4)
            
            col1.metric(label = "Avg Monthly ROI",
                        value = f"{return_risk['avg_monthly_roi']:.2f}%",
                        help  = "Average monthly return on investment",
                       )
            
            col2.metric(label = "ROI Volatility",
                        value = f"{return_risk['roi_volatility']:.2f}%",
                        help  = "Standard deviation of monthly ROI",
                       )
            
            col3.metric(label = "Sharpe Ratio",
                        value = f"{return_risk['sharpe_ratio']:.2f}",
                        help  = "Risk-adjusted return measure (higher is better)",
                       )
            
            col4.metric(label = "Max Drawdown",
                        value = f"{return_risk['max_drawdown_pct']:.1f}%",
                        help  = "Maximum peak-to-trough decline",
                       )
            
            # Risk assessment alert
            risk_level = return_risk['risk_assessment']
            
            if (risk_level == "HIGH RISK"):
                st.error(f"🚨 **{risk_level}** - Consider risk mitigation strategies")
            
            elif (risk_level == "MODERATE RISK"):
                st.warning(f"⚠️ **{risk_level}** - Monitor closely and optimize where possible")
            
            else:
                st.success(f"✅ **{risk_level}** - Returns are well-balanced")
            
            # Additional risk metrics
            col1, col2 = st.columns(spec = 2)
            
            col1.metric(label = "Value at Risk (5%)",
                        value = f"{return_risk['value_at_risk_5pct']:.2f}%",
                        help  = "5% worst-case monthly ROI",
                       )
            
            col2.metric(label = "Downside Deviation",
                        value = f"{return_risk['downside_deviation']:.2f}%",
                        help  = "Volatility of negative returns only",
                       )
            
            st.markdown("---")
            st.subheader(body = "📊 Profitability Trend Alerts")
            
            # Get profitability trend analysis
            trend_analysis = risk_eval.profitability_trend_alert()
            
            # Display alerts
            for alert in trend_analysis['alerts']:
                if ("🚨" in alert):
                    st.error(alert)
                
                elif ("⚠️" in alert):
                    st.warning(alert)

                else:
                    st.success(alert)
            
            # Display trend metrics
            if trend_analysis['trends']:
                st.write("**Trend Analysis Summary:**")
                
                col1, col2, col3 = st.columns(spec = 3)
                
                trends            = trend_analysis['trends']
                
                col1.metric(label = "Revenue Trend",
                            value = trends['revenue_trend'],
                            delta = f"{trends['revenue_change_pct']:.2f}%",
                            help  = f"Recent {trends['periods_analyzed']} periods vs previous periods",
                           )
                
                col2.metric(label = "Margin Trend", 
                            value = trends['margin_trend'],
                            delta = f"{trends['margin_change_pts']:.1f} pp",
                            help  = "Profit margin change in percentage points",
                           )
                
                col3.metric(label = "ROI Trend",
                            value = trends['roi_trend'],
                            delta = f"{trends['roi_change_pct']:.1f}%",
                            help  = "Return on investment trend",
                           )
                
                # Recent performance summary
                st.write("**Recent Performance Averages:**")
                perf_col1, perf_col2, perf_col3 = st.columns(spec = 3)
                
                perf_col1.metric(label = "Recent Avg Revenue",
                                 value = f"${trends['recent_avg_revenue']:,.0f}",
                                )
                
                perf_col2.metric(label = "Recent Avg Margin",
                                 value = f"{trends['recent_avg_margin']:.2f}%",
                                )
                
                perf_col3.metric(label = "Recent Avg ROI",
                                 value = f"{trends['recent_avg_roi']:.2f}%",
                                )
    
        
            st.markdown("---")
            st.subheader(body = "✅ Summary of Key Risk Indicators")
            st.markdown(risk_eval.generate_summary_text())
            
            st.markdown("---")
            st.subheader(body = "📌 Automated Insights")
            
            for insight in risk_eval.generate_risk_insights():
                st.markdown(insight)
            
            st.info("⚠️ Use these indicators to track business health and identify areas for action")
            
            # Download functionality for detailed analysis
            st.markdown("---")
            with st.expander("📥 Download Detailed Analysis", expanded = True):
                
                # Create comprehensive analysis dictionary
                comprehensive_analysis = {'low_margin_analysis'  : low_margin_analysis,
                                          'return_risk_analysis' : return_risk,
                                          'trend_analysis'       : trend_analysis,
                                          'basic_metrics'        : {'volatility'     : risk_eval.volatility,
                                                                    'hhi_country'    : risk_eval.hhi_country,
                                                                    'hhi_product'    : risk_eval.hhi_product,
                                                                    'low_margin_pct' : risk_eval.low_margin_pct,
                                                                    'perf_decline'   : risk_eval.perf_decline,
                                                                   }
                                         }
                
                st.json(comprehensive_analysis, expanded = False)
                
                # If you want to offer CSV downloads of the low-margin transactions
                if (not low_margin_analysis['low_margin_transactions'].empty):
                    csv_data = low_margin_analysis['low_margin_transactions'].to_csv(index = False)

                    st.download_button(label     = "📄 Download Low-Margin Transactions CSV",
                                       data      = csv_data,
                                       file_name = f"low_margin_transactions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                       mime      = "text/csv",
                                      )


        with st.expander(label = "🚨 Business Alerts & Real-Time Risk Dashboard", expanded = True):
            st.subheader(body = "🚨 Live Business Health Alerts & Risk Overview")
        
            risk_evaluator = BusinessRiskEvaluator(filtered_data)
        
            # Monthly Revenue Change
            latest_rev, change = risk_evaluator.compute_latest_monthly_change()
            
            if (latest_rev is not None):
                st.metric(label       = "Latest Monthly Revenue", 
                          value       = f"${latest_rev:,.0f}", 
                          delta       = f"{change:.1f}%",
                          delta_color = "inverse" if change < 0 else "normal",
                         )
        
                if (change < -20):
                    st.error(f"🔴 **Critical Drop**: Revenue fell {change:.1f}% since last month")
                
                elif (change < -10):
                    st.warning(f"🟡 **Warning**: Revenue down {change:.1f}% from previous month")
                
                else:
                    st.success(f"✅ Stable/Positive Revenue Change: {change:.1f}%")
                
                st.line_chart(risk_evaluator.monthly_revenue)
                
            else:
                st.info("Not enough data to compute monthly revenue change")
        
            st.markdown("---")
            st.subheader(body = "📊 Quick Risk KPIs")
        
            col1, col2, col3 = st.columns(spec = 3)
        
            hhi              = risk_evaluator.compute_market_concentration_hhi()
            low_margin_pct   = risk_evaluator.compute_low_margin_pct()
            rev_cv           = risk_evaluator.compute_revenue_volatility_cv()
        
            col1.metric(label = "Market Concentration (HHI)", 
                        value = f"{hhi:.0f}",
                       )
            
            col1.progress(min(hhi / 3000, 1.0))
        
            col2.metric(label = "Low Margin Transactions (%)", 
                        value = f"{low_margin_pct:.1f}%",
                       )
            
            col2.progress(min(low_margin_pct / 100, 1.0))
        
            if rev_cv is not None:
                col3.metric(label = "Revenue Volatility (CV)", 
                            value = f"{rev_cv:.1f}%",
                           )
                
                col3.progress(min(rev_cv / 50, 1.0))
                
            else:
                col3.info("Not enough data for CV")

            st.caption("""
            🟢 Green = Healthy | 🟡 Caution Zone | 🔴 Red = High Risk  
            📌 Ideal Benchmarks → HHI < 1500, Low Margin < 25%, Revenue CV < 20%
            """)
            
            # 📝 Automated Insights Section
            st.markdown(body = "### 📌 Automated Summary Insights")
            
            insights = list()
            
            # HHI Analysis
            if (hhi > 2500):
                insights.append("🔴 **High Market Concentration Detected (HHI > 2500)** → Risk of overdependence on few markets")
            
            elif (hhi > 1500):
                insights.append("🟡 **Moderate Market Concentration (HHI between 1500-2500)** → Diversification advisable")
            
            else:
                insights.append("🟢 **Low Market Concentration (HHI < 1500)** → Healthy market spread")
            
            # Low Margin Risk Analysis
            if (low_margin_pct > 40):
                insights.append("🔴 **High Exposure to Low-Margin Sales (>40%)** → Urgent profitability improvement needed")
            
            elif (low_margin_pct > 25):
                insights.append("🟡 **Moderate Low-Margin Exposure (25%-40%)** → Monitor margins closely")
            
            else:
                insights.append("🟢 **Healthy Profit Margins (Low-Margin <25%)** → Profitability within safe limits")
            
            # Revenue Volatility (CV) Analysis
            if (rev_cv is not None):
                if (rev_cv > 30):
                    insights.append("🔴 **High Revenue Volatility (CV > 30%)** → Revenue stability risks present")
                
                elif (rev_cv > 20):
                    insights.append("🟡 **Moderate Revenue Volatility (CV 20%-30%)** → Track trends for anomalies")
                
                else:
                    insights.append("🟢 **Stable Revenue (CV < 20%)** → Predictable revenue performance")
            
            else:
                insights.append("ℹ️ **Not enough data points to compute revenue volatility insights**")

            # Display Insights
            for insight in insights:
                st.markdown(insight)
            
            # 📦 Support Cost Allocation Model
            st.markdown("---")
            st.subheader(body = "📦 Support Cost Allocation Model")

            support_costs_df = risk_evaluator.calculate_support_cost_allocation()

            fig              = px.bar(data_frame = support_costs_df,
                                      x          = "Product",
                                      y          = "Cost_per_Unit",
                                      color      = "Cost_per_Unit",
                                      title      = "Cost per Unit by Product (Allocate Support Resources)",
                                     )

            st.plotly_chart(figure_or_data      = fig, 
                            use_container_width = True,
                           )


        with st.expander(label = "📜 Executive Summary & Recommendations", expanded = True):
            summary_gen = ExecutiveSummaryGenerator(filtered_df = filtered_data)
            
            summary_gen.display_summary()

    except Exception as e:
        raise
        #st.error(f"Error: {e}")
else:
    st.info("Please upload a sales dataset to begin the analysis")
