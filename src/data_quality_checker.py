# DEPENDENCIES
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from datetime import timedelta


# SANITY CHECKING
class DataQuality:
    def __init__(self, dataframe):
        self.df = dataframe
    
        # Convert date column if it exists and is numeric (Excel date format)
        if (('Date' in self.df.columns) and (pd.api.types.is_numeric_dtype(self.df['Date']))):
            try:
                # Convert Excel serial date to datetime
                self.df['Date'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(self.df['Date'] - 2, unit='D')
            
            except:
                # If conversion fails, try to parse as regular datetime
                self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        elif 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
    

    def generate_report(self):
        rows, cols     = self.df.shape
        total_missing  = self.df.isnull().sum().sum()
        duplicates     = self.df.duplicated().sum()
        product_count  = self.df['Product'].nunique() if 'Product' in self.df.columns else None
        country_count  = self.df['Country'].nunique() if 'Country' in self.df.columns else None

        # Unified variable summary
        column_summary = list()
    
        for column in self.df.columns:
            dtype         = str(self.df[column].dtype)
            missing_count = self.df[column].isnull().sum()
            outlier_count = None
            
            if (np.issubdtype(self.df[column].dtype, np.number)):
                z_scores      = np.abs(stats.zscore(self.df[column].dropna()))
                outlier_count = (z_scores > 3).sum()
            
            column_summary.append({'Column'        : column,
                                   'Data Type'     : dtype,
                                   'Missing Count' : missing_count,
                                   'Outlier Count' : outlier_count,
                                 })
    
        column_summary_df = pd.DataFrame(data = column_summary)

        # Numeric Summary
        numeric_columns   = self.df.select_dtypes(include = [np.number]).columns
        summary_stats     = self.df[numeric_columns].describe().T.round(2)

        # Consolidate summary in a dictionary
        qa_report         =  {'shape'            : (rows, cols),
                              'missing_total'    : total_missing,
                              'duplicates'       : duplicates,
                              'unique_products'  : product_count,
                              'unique_countries' : country_count,
                              'column_summary'   : column_summary_df,
                              'summary_stats'    : summary_stats,
                             }
        return qa_report

    
    def detect_business_logic_anomalies(self):
        """
        Detect business logic anomalies specific to sales data
        
        Returns:
            { dict } : Dictionary containing various business logic anomaly checks
        """
        anomalies        = {'negative_values'           : {},
                            'zero_values'               : {},
                            'inconsistent_calculations' : {},
                            'extreme_ratios'            : {},
                            'date_anomalies'            : {},
                            'summary_flags'             : [],
                           }
        
        # Check for negative values where they shouldn't exist
        business_columns = ['Revenue', 
                            'Cost', 
                            'Profit', 
                            'Units Sold',
                           ]

        for col in business_columns:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                
                if (negative_count > 0):
                    anomalies['negative_values'][col] = {'count'      : negative_count,
                                                         'percentage' : round((negative_count / len(self.df)) * 100, 2),
                                                         'examples'   : self.df[self.df[col] < 0][col].head().tolist(),
                                                        }
        
        # Check for zero values in critical columns
        critical_columns = ['Revenue', 
                            'Units Sold',
                           ]

        for col in critical_columns:
            if (col in self.df.columns):
                zero_count = (self.df[col] == 0).sum()
                
                if (zero_count > 0):
                    anomalies['zero_values'][col] = {'count'      : zero_count,
                                                     'percentage' : round((zero_count / len(self.df)) * 100, 2),
                                                    }
        
        # Check calculation consistency (Revenue = Units Sold × Unit Price, Profit = Revenue - Cost)
        if (all(col in self.df.columns) for col in ['Revenue', 'Units Sold']):
            # Calculate implied unit price
            valid_mask = (self.df['Units Sold'] > 0) & (self.df['Revenue'] > 0)
            
            if (valid_mask.sum() > 0):
                unit_prices = self.df.loc[valid_mask, 'Revenue'] / self.df.loc[valid_mask, 'Units Sold']
                price_cv    = (unit_prices.std() / unit_prices.mean()) * 100 if unit_prices.mean() > 0 else 0
                
                anomalies['inconsistent_calculations']['unit_price_variation'] = {'coefficient_variation' : round(price_cv, 2),
                                                                                  'min_price'             : round(unit_prices.min(), 2),
                                                                                  'max_price'             : round(unit_prices.max(), 2),
                                                                                  'median_price'          : round(unit_prices.median(), 2),
                                                                                 }
        
        if (all(col in self.df.columns) for col in ['Revenue', 'Cost', 'Profit']):
            # Check if Profit = Revenue - Cost
            calculated_profit         = self.df['Revenue'] - self.df['Cost']
            profit_discrepancy        = abs(self.df['Profit'] - calculated_profit)

            # Allow for small rounding errors (< 0.01)
            significant_discrepancies = (profit_discrepancy > 0.01).sum()
            
            anomalies['inconsistent_calculations']['profit_calculation'] = {'discrepancy_count' : significant_discrepancies,
                                                                            'percentage'        : round((significant_discrepancies / len(self.df)) * 100, 2),
                                                                            'max_discrepancy'   : round(profit_discrepancy.max(), 2),
                                                                           }
        
        # Check for extreme ratios/margins
        if (all(col in self.df.columns) for col in ['Revenue', 'Cost']):
            # Calculate profit margins
            valid_revenue = self.df['Revenue'] > 0
            
            if (valid_revenue.sum() > 0):
                margins         = ((self.df.loc[valid_revenue, 'Revenue'] - self.df.loc[valid_revenue, 'Cost']) / self.df.loc[valid_revenue, 'Revenue']) * 100
                
                extreme_margins = {'negative_margins'   : (margins < 0).sum(),
                                   'super_high_margins' : (margins > 95).sum(),
                                   'margins_over_100'   : (margins > 100).sum(),
                                  }
                
                anomalies['extreme_ratios']['profit_margins'] = extreme_margins
        
        # Date-related anomalies
        if ('Date' in self.df.columns):
            date_col       = pd.to_datetime(self.df['Date'], errors='coerce')
            
            # Future dates
            today          = pd.Timestamp.now()
            future_dates   = (date_col > today).sum()
            
            # Very old dates (more than 10 years ago)
            ten_years_ago  = today - pd.DateOffset(years=10)
            very_old_dates = (date_col < ten_years_ago).sum()
            
            # Date gaps (missing months/quarters)
            if not date_col.isnull().all():
                date_range                  = pd.date_range(start=date_col.min(), end=date_col.max(), freq='M')
                monthly_counts              = date_col.dt.to_period('M').value_counts()
                missing_months              = len(date_range) - len(monthly_counts)
                
                anomalies['date_anomalies'] = {'future_dates'     : future_dates,
                                               'very_old_dates'   : very_old_dates,
                                               'missing_months'   : missing_months,
                                               'date_range_years' : round((date_col.max() - date_col.min()).days / 365.25, 1),
                                              }
        
        # Generate summary flags
        if (anomalies['negative_values']):
            anomalies['summary_flags'].append("⚠️ Negative values detected in business metrics")
        
        if (anomalies['zero_values']):
            anomalies['summary_flags'].append("⚠️ Zero values found in critical columns")
        
        if (anomalies['inconsistent_calculations']):
            profit_issues = anomalies['inconsistent_calculations'].get('profit_calculation', {})
            
            if (profit_issues.get('discrepancy_count', 0) > 0):
                anomalies['summary_flags'].append("❗ Profit calculation inconsistencies detected")
        
        if (not anomalies['summary_flags']):
            anomalies['summary_flags'].append("✅ No major business logic anomalies detected")
        
        return anomalies


    def validate_data_completeness(self, required_columns = None, date_range_check = True):
        """
        Validate data completeness and identify gaps
        
        Arguments:
        ----------
            required_columns { list } : List of columns that must be present and complete

            date_range_check { bool } : Whether to check for date continuity
        
        Returns:
        --------
                    { dict }          : Completeness validation results
        """
        if required_columns is None:
            required_columns = ['Date', 
                                'Country', 
                                'Product', 
                                'Revenue', 
                                'Units Sold',
                               ]
        
        completeness                    = {'column_presence'       : {},
                                           'missing_data_analysis' : {},
                                           'date_continuity'       : {},
                                           'completeness_score'    : 0,
                                           'recommendations'       : [],
                                          }
        
        # Check column presence
        missing_cols                    = [col for col in required_columns if col not in self.df.columns]
        present_cols                    = [col for col in required_columns if col in self.df.columns]
        
        completeness['column_presence'] = {'required_columns' : required_columns,
                                           'missing_columns'  : missing_cols,
                                           'present_columns'  : present_cols,
                                           'presence_rate'    : round((len(present_cols) / len(required_columns)) * 100, 1),
                                          }
        
        # Analyze missing data patterns
        for col in present_cols:
            missing_count                              = self.df[col].isnull().sum()
            missing_pct                                = (missing_count / len(self.df)) * 100
            
            completeness['missing_data_analysis'][col] = {'missing_count'           : missing_count,
                                                          'missing_percentage'      : round(missing_pct, 2),
                                                          'completeness_percentage' : round(100 - missing_pct, 2),
                                                         }
        
        # Date continuity analysis
        if (date_range_check and ('Date' in self.df.columns)):
            date_col = pd.to_datetime(self.df['Date'], errors = 'coerce')
            
            if (not date_col.isnull().all()):
                # Check for gaps in monthly data
                monthly_data = date_col.dt.to_period('M').value_counts().sort_index()
                
                if (len(monthly_data) > 1):
                    expected_months                 = pd.period_range(start = monthly_data.index.min(), 
                                                                      end   = monthly_data.index.max(), 
                                                                      freq  = 'M',
                                                                     )
                    
                    missing_months                  = set(expected_months) - set(monthly_data.index)
                    
                    completeness['date_continuity'] = {'total_months_expected' : len(expected_months),
                                                       'total_months_present'  : len(monthly_data),
                                                       'missing_months_count'  : len(missing_months),
                                                       'missing_months_list'   : [str(month) for month in sorted(missing_months)],
                                                       'continuity_percentage' : round((len(monthly_data) / len(expected_months)) * 100, 1),
                                                      }
        
        # Calculate overall completeness score
        col_presence_score  = (len(present_cols) / len(required_columns)) * 40
        
        missing_data_scores = list()

        for col in present_cols:
            if col in completeness['missing_data_analysis']:
                missing_data_scores.append(completeness['missing_data_analysis'][col]['completeness_percentage'])
        
        avg_completeness        = sum(missing_data_scores) / len(missing_data_scores) if missing_data_scores else 0
        data_completeness_score = (avg_completeness / 100) * 40
        
        date_continuity_score   = 0
        
        if (completeness['date_continuity']):
            date_continuity_score = (completeness['date_continuity']['continuity_percentage'] / 100) * 20
        
        else:
            # Full score if no date check required
            date_continuity_score = 20 
        
        completeness['completeness_score'] = round(col_presence_score + data_completeness_score + date_continuity_score, 1)
        
        # Generate recommendations
        if missing_cols:
            completeness['recommendations'].append(f"❗ Add missing columns: {', '.join(missing_cols)}")
        
        high_missing_cols = [col for col, data in completeness['missing_data_analysis'].items() if data['missing_percentage'] > 10]
        
        if high_missing_cols:
            completeness['recommendations'].append(f"⚠️ Address high missing data in: {', '.join(high_missing_cols)}")
        
        if (completeness['date_continuity'] and (completeness['date_continuity']['missing_months_count'] > 0)):
            completeness['recommendations'].append("⚠️ Fill gaps in date continuity for better trend analysis")
        
        if (completeness['completeness_score'] >= 90):
            completeness['recommendations'].append("✅ Excellent data completeness!")
        
        elif (completeness['completeness_score'] >= 75):
            completeness['recommendations'].append("✅ Good data completeness with minor issues")
        
        else:
            completeness['recommendations'].append("❗ Data completeness needs improvement")
        
        return completeness

    
    def generate_data_profiling_summary(self):
        """
        Generate a comprehensive data profiling summary
        
        Returns:
        --------
            { dict } : Comprehensive data profiling results
        """
        profiling                     = {'dataset_overview'       : {},
                                         'categorical_analysis'   : {},
                                         'numerical_analysis'     : {},
                                         'relationships_analysis' : {},
                                         'data_quality_score'     : 0,
                                        }
        
        # Dataset overview
        profiling['dataset_overview'] = {'total_records'   : len(self.df),
                                         'total_columns'   : len(self.df.columns),
                                         'memory_usage_mb' : round(self.df.memory_usage(deep=True).sum() / (1024*1024), 2),
                                         'date_range'      : None,
                                         'primary_metrics' : {},
                                        }
        
        # Add date range if available
        if ('Date' in self.df.columns):
            date_col = pd.to_datetime(self.df['Date'], errors='coerce')
            
            if (not date_col.isnull().all()):
                profiling['dataset_overview']['date_range'] = {'start_date'   : str(date_col.min().date()),
                                                               'end_date'     : str(date_col.max().date()),
                                                               'days_covered' : (date_col.max() - date_col.min()).days,
                                                              }
        
        # Primary business metrics
        metric_cols = ['Revenue', 
                       'Profit', 
                       'Units Sold',
                      ]

        for col in metric_cols:
            if col in self.df.columns:
                profiling['dataset_overview']['primary_metrics'][col] = {'total'   : round(self.df[col].sum(), 2),
                                                                         'average' : round(self.df[col].mean(), 2),
                                                                         'min'     : round(self.df[col].min(), 2),
                                                                         'max'     : round(self.df[col].max(), 2),
                                                                        }
        
        # Categorical analysis
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:

            # Skip date if it's still object type
            if (col != 'Date'):  
                value_counts                           = self.df[col].value_counts()
                
                profiling['categorical_analysis'][col] = {'unique_values'         : self.df[col].nunique(),
                                                          'most_frequent'         : value_counts.index[0] if len(value_counts) > 0 else None,
                                                          'most_frequent_count'   : value_counts.iloc[0] if len(value_counts) > 0 else 0,
                                                          'distribution_evenness' : round(1 - (value_counts.std() / value_counts.mean()), 3) if len(value_counts) > 1 else 1,
                                                          'top_5_values'          : value_counts.head().to_dict(),
                                                         }
        
        # Numerical analysis with business context
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Skip date if it's numeric
            if (col != 'Date'):  
                col_data = self.df[col].dropna()
                
                if (len(col_data) > 0):
                    profiling['numerical_analysis'][col] = {'mean'                  : round(col_data.mean(), 2),
                                                            'median'                : round(col_data.median(), 2),
                                                            'std_dev'               : round(col_data.std(), 2),
                                                            'coefficient_variation' : round((col_data.std() / col_data.mean()) * 100, 2) if col_data.mean() != 0 else 0,
                                                            'skewness'              : round(col_data.skew(), 3),
                                                            'kurtosis'              : round(col_data.kurtosis(), 3),
                                                            'outlier_percentage'    : round(((np.abs(stats.zscore(col_data)) > 3).sum() / len(col_data)) * 100, 2)
                                                           }
        
        # Relationships analysis
        if (all(col in self.df.columns) for col in ['Revenue', 'Cost']):
            correlation                                                     = self.df[['Revenue', 'Cost']].corr().iloc[0, 1]
            profiling['relationships_analysis']['revenue_cost_correlation'] = round(correlation, 3)
        
        if (all(col in self.df.columns) for col in ['Revenue', 'Units Sold']):
            correlation                                                      = self.df[['Revenue', 'Units Sold']].corr().iloc[0, 1]
            profiling['relationships_analysis']['revenue_units_correlation'] = round(correlation, 3)
        
        # Calculate data quality score
        basic_report                    = self.generate_report()
        anomalies                       = self.detect_business_logic_anomalies()
        completeness                    = self.validate_data_completeness()
        
        # Scoring components (out of 100)
        missing_data_score              = max(0, 100 - (basic_report['missing_total'] / (len(self.df) * len(self.df.columns))) * 100)
        duplicate_score                 = max(0, 100 - (basic_report['duplicates'] / len(self.df)) * 100)
        anomaly_score                   = max(0, 100 - len([flag for flag in anomalies['summary_flags'] if '⚠️' in flag or '❗' in flag]) * 20)
        completeness_score              = completeness['completeness_score']
        
        profiling['data_quality_score'] = round((missing_data_score + duplicate_score + anomaly_score + completeness_score) / 4, 1)
        
        return profiling
