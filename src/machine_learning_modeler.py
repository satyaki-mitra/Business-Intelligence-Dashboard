# DEPENDENCIES
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


# MACHINE LEARNING MODELS
class MLModelEvaluator:
    def __init__(self, dataframe, target, features, scale_data = False, random_state = 42, test_size = 0.2):
        """
        Robust ML Model Evaluator with scaling and persistence
        """
        self.df            = dataframe.copy()
        self.target        = target
        self.features      = features
        self.scale_data    = scale_data
        self.random_state  = random_state
        self.test_size     = test_size

        self.scaler        = None    # will store fitted scaler
        self.fitted_models = dict()  # store trained models
        
        self._validate_inputs()
        self._prepare_data()

    
    def _validate_inputs(self):
        missing_cols = [col for col in self.features + [self.target] if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Convert datetime columns to timestamp
        for column in self.df.select_dtypes(include = ['datetime64[ns]', 'datetime64[ns, UTC]']):
            self.df[column] = self.df[column].astype('int64') // 10 ** 9

        # Simple NaN handling (you can customize)
        self.df.fillna(0, inplace = True)

    
    def _prepare_data(self):
        X = self.df[self.features].copy()
        y = self.df[self.target]

        if self.scale_data:
            self.scaler = StandardScaler()
            X           = pd.DataFrame(self.scaler.fit_transform(X), columns = self.features)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
                                                                                y, 
                                                                                random_state = self.random_state, 
                                                                                test_size    = self.test_size,
                                                                               )
        

    def evaluate_model(self, model, name):
        try:
            model.fit(self.X_train, 
                      self.y_train,
                     )
            
            preds = model.predict(self.X_test)

            r2    = r2_score(self.y_test, preds)
            mse   = mean_squared_error(self.y_test, preds)
            rmse  = np.sqrt(mse)
            cv_r2 = cross_val_score(model, 
                                    self.X_train, 
                                    self.y_train, 
                                    cv      = 5, 
                                    scoring = 'r2').mean()

            # store the trained model
            self.fitted_models[name] = model  

            return {'model' : model,
                    'r2'    : round(r2, 3),
                    'rmse'  : round(rmse, 2),
                    'cv_r2' : round(cv_r2, 3),
                    'preds' : preds,
                    'name'  : name,
                   }
            
        except Exception as e:
            return {'model' : None, 
                    'r2'    : None, 
                    'rmse'  : None, 
                    'cv_r2' : None, 
                    'preds' : None,
                    'name'  : name, 
                    'error' : str(e),
                   }

    
    def feature_importance_plot(self, model):
        if (hasattr(model, "feature_importances_")):
            importances   = model.feature_importances_
            df_importance = pd.DataFrame({'Feature'    : self.features, 
                                          'Importance' : importances,
                                        })
            
            fig           = px.bar(data_frame  = df_importance.sort_values(by        = 'Importance', 
                                                                           ascending = False,
                                                                          ),
                                   x           = 'Importance', 
                                   y           = 'Feature', 
                                   orientation = 'h',
                                   title       = '🔎 Feature Importance',
                                  )
            return fig

        elif (hasattr(model, "coef_")):
            importances   = np.abs(model.coef_)
            df_importance = pd.DataFrame({'Feature': self.features, 'Importance': importances})
            fig           = px.bar(data_frame       = df_importance.sort_values(by        = 'Importance', 
                                                                                ascending = False,
                                                                               ),
                                   x                = 'Importance', 
                                   y                = 'Feature', 
                                   orientation      = 'h',
                                   title            = '🔎 Feature Importance (Linear Coefficients)')
            return fig

        return None

    
    def predict_new_data(self, new_data, model_name=None):
        """
        Predict on new unseen data using already fitted model
        """
        X_new = new_data[self.features].copy()

        if self.scale_data and self.scaler is not None:
            X_new = pd.DataFrame(self.scaler.transform(X_new), 
                                 columns = self.features,
                                )

        if (model_name is None):
            raise ValueError("Please specify model_name to use for prediction")

        if (model_name not in self.fitted_models):
            raise ValueError(f"Model '{model_name}' not trained yet. Train it using evaluate_model()")

        model = self.fitted_models[model_name]
        preds = model.predict(X_new)
        
        return preds
 