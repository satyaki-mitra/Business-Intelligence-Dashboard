# DEPENDENCIES
import pandas as pd
from datetime import datetime


# DATA LOADING 
class DataLoader:
    def __init__(self, uploaded_file):
        self.uploaded_file  = uploaded_file
        self.sales_data_raw = None

    
    def load_data(self):
        """
        Load CSV or Excel data into pandas dataframe
        """
        if (self.uploaded_file is None):
            raise ValueError("No file uploaded")

        file_extension = self.uploaded_file.name.split('.')[-1].lower()

        if (file_extension == 'csv'):
            self.sales_data_raw = pd.read_csv(filepath_or_buffer = self.uploaded_file)
        
        elif (file_extension == 'xlsx'):
            self.sales_data_raw = pd.read_excel(io = self.uploaded_file)
        
        else:
            raise ValueError("Unsupported file format. Please upload CSV or XLSX.")

        return self.sales_data_raw

    @staticmethod
    def convert_excel_date(excel_date):
        """
        Convert Excel numeric date to datetime object
        """
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(excel_date, unit = 'D')


    @staticmethod
    def _covid_period(x):
        
        if (x < datetime(2020, 3, 1)):
            return "Pre-COVID"
            
        elif (x < datetime(2020, 6, 1)):
            return "COVID-Impact"
            
        else:
            return "Recovery"
            
    
    def preprocess_data(self):
        """
        Preprocess sales data: date conversion, profit margin calculation, COVID tagging
        """
        if self.sales_data_raw is None:
            raise ValueError("Data not loaded. Run load_data() first")

        # Date conversion
        self.sales_data_raw['Date']          = self.sales_data_raw['Date'].apply(self.convert_excel_date)

        # Profit margin calculation
        self.sales_data_raw['Profit_Margin'] = ((self.sales_data_raw['Profit'] / self.sales_data_raw['Revenue']) * 100)

        # COVID Period tagging
        self.sales_data_raw['COVID_Period']  = self.sales_data_raw['Date'].apply(self._covid_period)

        return self.sales_data_raw

