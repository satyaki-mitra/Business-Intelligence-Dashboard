# DEPENDENCIES
import pandas as pd
import streamlit as st


# EXECUTIVE SUMMARY GENERATOR
class ExecutiveSummaryGenerator:
    def __init__(self, filtered_df):
        self.df = filtered_df

    
    def get_key_metrics(self):
        df      = self.df
        
        summary = {"top_market"              : df.groupby('Country')['Revenue'].sum().idxmax(),
                   "top_market_revenue"      : df.groupby('Country')['Revenue'].sum().max(),
                   "top_product"             : df.groupby('Product')['Revenue'].sum().idxmax(),
                   "top_product_revenue"     : df.groupby('Product')['Revenue'].sum().max(),
                   "most_profitable_product" : df.groupby('Product')['Profit_Margin'].mean().idxmax(),
                   "highest_margin"          : df.groupby('Product')['Profit_Margin'].mean().max(),
                   "total_transactions"      : len(df),
                   "avg_transaction_value"   : df['Revenue'].mean(),
                   "market_count"            : df['Country'].nunique(),
                   "product_count"           : df['Product'].nunique()
                  }
        
        return summary

    
    def display_summary(self):
        summary = self.get_key_metrics()

        st.subheader(body = "📜 Executive Summary Highlights")

        st.markdown(f"""
        ### 🔑 **Key Highlights**
        - 🏆 **Top Market by Revenue:** **{summary['top_market']}** (${summary['top_market_revenue']:,.0f})
        - 🥇 **Top Product by Revenue:** **{summary['top_product']}** (${summary['top_product_revenue']:,.0f})
        - 💰 **Most Profitable Product (Avg Margin):** **{summary['most_profitable_product']}** ({summary['highest_margin']:.1f}% Profit Margin)
        - 📊 **Total Transactions:** **{summary['total_transactions']:,}**
        - 💵 **Avg Transaction Value:** **${summary['avg_transaction_value']:,.0f}**
        - 🌎 **Markets Covered:** **{summary['market_count']} Countries**
        - 📦 **Products Portfolio:** **{summary['product_count']} Unique Products**
        """)

        st.markdown("---")
        st.subheader(body = "🎯 Strategic Recommendations")

        col1, col2 = st.columns(spec = 2)
        
        with col1:
            st.markdown(body = "#### 📌 **Market Strategy**")
            st.markdown(f"""
            - 📈 **Focus Market Expansion** in **{summary['top_market']}** to further boost high revenue performance.
            - 🌍 **Cross-Learnings:** Apply successful strategies from **{summary['top_market']}** to underperforming markets.
            - 🛍️ **Optimize Pricing Strategy** in moderate performing countries to maximize margin.
            """)

        with col2:
            st.markdown(body = "#### 📌 **Product Strategy**")
            st.markdown(f"""
            - 🚀 **Boost Sales of {summary['top_product']}**, the top revenue product, with targeted promotions.
            - 🟢 **Enhance Margins on {summary['most_profitable_product']}**, your highest margin product.
            - ⚙️ **Balanced Product Mix:** Combine volume drivers (**{summary['top_product']}**) and margin drivers (**{summary['most_profitable_product']}**) in campaigns.
            - 🎁 **Seasonal Product Development:** Explore new launches based on seasonal peaks.
            """)

        st.markdown("---")
        st.subheader(body = "✅ **Operational Focus**")
        st.markdown(f"""
        - 🚚 **Supply Chain Priority:** Prioritize stock allocation for high-volume markets like **{summary['top_market']}**.
        - 📦 **Inventory Efficiency:** Evaluate slow movers for optimization or replacement.
        - 🖥️ **Digital Channels Expansion:** Leverage e-commerce channels in high-performing markets.
        """)
        st.success("📢 **Actionable Recommendations Generated Based on Current Data Filters**")

