# 📊 Sales Performance & Business Intelligence Dashboard

## 🚀 Project Overview
This interactive, enterprise-grade **Sales Performance Dashboard** provides end-to-end analytics for a specialty product company operating across multi-country markets. Built with **Python** and **Streamlit**, it delivers:

- 📈 Sales insights, profitability analysis, and risk monitoring
- 🤖 AI-powered forecasts and machine learning modeling
- 📝 Boardroom-ready executive summaries and strategic recommendations

---

## 📝 Business Objectives
- ✅ Identify top-performing **markets** and **products**
- ✅ Detect **profitability risks**, **volatility**, and **performance shifts**
- ✅ Forecast future sales using **time-series models**
- ✅ Compare **machine learning models** for predictive capabilities
- ✅ Deliver **actionable insights** through a fully automated **executive summary**

---

## 📂 Project Structure
```bash
📁 sales_performance_dashboard/
│
├── 📜 README.md                          # Project Documentation
├── 📁 data/                              # Raw and processed datasets
│
├── 📁 src/                               # Core Streamlit App Modules
│   ├── business_risk_evaluator.py        # Business health monitoring and alerts
│   ├── data_loader.py                    # Data ingestion and preprocessing logic
│   ├── data_quality_checker.py           # Data quality and data integrity reporting
│   ├── executive_summary_helper.py       # Auto-generated summary for executives
│   ├── exploratory_data_analyzer.py      # Interactive EDA with product and market filters
│   ├── machine_learning_modeler.py       # Regression modeling, feature importance, prediction
│   ├── market_segmentation_summarizer.py # Market segmentation and analysis
│   ├── risk_analyzer.py                  # Risk analysis (volatility, concentration, low margin)
│   ├── statistical_tester.py             # Hypothesis testing module (ANOVA, Shapiro, etc.)
│   └── time_series_analyzer.py           # Forecasting with ARIMA, Holt-Winters, etc.
│
├── dashboard_app.py                      # 📊 Main Streamlit Application Entry Point
└── requirements.txt                      # Python package dependencies
```

---

## 💻 Core Functionalities & Flow

| Module                          | Purpose                                                                 |
|---------------------------------|-------------------------------------------------------------------------|
| Business Risk Evaluator         | Live health monitoring with delta analysis and automated warnings.     |
| Data Loader                     | Load and preprocess raw transactional sales data.                      |
| Data Quality Checker            | Analyze data integrity, missing data, outliers, duplicates, and basic statistics. |
| Executive Summary Helper        | Auto-generated key findings and actionable business strategy.          |
| Exploratory Data Analyzer       | Dynamic filters to analyze KPIs by date, country, product.              |
| Machine Learning Modeler        | Train/test multiple regression models, assess R², RMSE, CV scores.     |
| Market Segmentation Summarizer  | Market segmentation and analysis.                                       |
| Risk Analyzer                   | Revenue volatility, HHI concentration risks, low margin exposure KPIs. |
| Statistical Tester               | Hypothesis testing module (normality, variance, group differences).    |
| Time Series Analyzer            | Monthly revenue trend analysis, seasonality decomposition, forecasts.  |


---

## 🏆 Key Features

- ✅ 📈 Fully dynamic dashboards using **Streamlit**
- ✅ 🖥️ Business-friendly **Plotly** visualizations
- ✅ 🤖 Predictive modeling with regression algorithms (**XGBoost**, **RandomForest**, **Lasso**, etc.)
- ✅ 🧮 Advanced statistical testing (**Shapiro-Wilk**, **ANOVA**, **Kruskal-Wallis**)
- ✅ 📝 Auto-generated **executive summary** with real-time strategy suggestions
- ✅ 🛡️ **Risk dashboards**: volatility, low-margin exposure, market concentration
- ✅ 💼 Suitable for **C-level dashboards** and **operational teams**

---

## 🧰 Tech Stack

| Component              | Details                                              |
|-------------------------|------------------------------------------------------|
| **Programming Language** | Python 3.9+                                         |
| **Data Visualization**   | Plotly, Seaborn, Matplotlib                          |
| **Web App Framework**    | Streamlit                                           |
| **Machine Learning**     | Scikit-learn, XGBoost                               |
| **Statistical Analysis** | Scipy, Statsmodels                                  |
| **Forecasting**          | Statsmodels (ARIMA, Holt-Winters)                   |
| **Deployment Ready**     | Streamlit Cloud / Docker                             |

---

## 📦 Setup Instructions

```bash
git clone https://github.com/your-repo/sales_performance_dashboard.git
cd sales_performance_dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit App
streamlit run dashboard_app.py
```

## 🎁 Deployment Options

- ✅ **Local Streamlit App**
- ✅ **Streamlit Cloud** (push to GitHub, connect to Streamlit Cloud)
- ✅ **Dockerization** for containerized deployments
- ✅ **Enterprise Cloud Options** (AWS, GCP, Azure)

---

## 📌 Future Expansion Ideas

- 🟢 **Email Alerts & Scheduled Reports**
- 🟢 **PDF Export of Executive Summary**
- 🟢 **More Robust Predictive Models** (LightGBM, CatBoost)
- 🟢 **Live API Integration for Real-Time Data**

---

## 📢 Contributors

| Name       | Role                          |
|-------------|-------------------------------|
| Your Name   | Data Scientist & Developer    |

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
