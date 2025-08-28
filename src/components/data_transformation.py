import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()


       
    def get_preprocessor(self, df:pd.DataFrame) -> ColumnTransformer:

       """
       This function Creates a preprocessing pipeline for credit card fraud detection dataset- 
            - Scales 'Amount' (after median imputation)
            - Passes PCA features (V1-V28) unchanged
            - Drops any other columns (like 'Time')

        """
       
       try:
          logging.info("Creating preprocessing pipeline for CCFD dataset")

          # Identify columns
          numerical_features= ['Amount']
          pca_features= [col for col in df.columns if col.startswith('V')]
          categorical_features= []

          #Pipelines
          num_pipeline= Pipeline(steps=[
              ('imputer', SimpleImputer(strategy='median')),
              ('scaler', StandardScaler())
          ])

          preprocessor= ColumnTransformer(transformers=[
              ('num',num_pipeline,numerical_features),
              ('pca', 'passthrough',pca_features),
          ],
          remainder='drop'
          )

          return preprocessor
       
       except Exception as e:
          logging.error("Error in creating preprocessor")
          raise CustomException(e,sys)
       

    def initiate_data_transformation(self, train_path:str, test_path:str):

        """
        This function-
            - Reads train/test CSVs, applies transformations, and saves preprocessor.
            - Returns transformed arrays and target arrays.

        """

        try:

            logging.info("Starting data transformation")
            
            # Read datasets
            df_train= pd.read_csv(train_path)
            df_test= pd.read_csv(test_path)
            logging.info('Train and Test data read successfully')

            # Drop 'Time' and separate target variable
            target= 'Class'

            #Train dataset
            X_train= df_train.drop(columns=['Time', 'Class'], axis=1)
            y_train= df_train[target]

            #Test dataset
            X_test= df_test.drop(columns=['Time', 'Class'], axis=1)
            y_test= df_test['Class']

            # Get preprocessor
            preprocessor= self.get_preprocessor(X_train)

            #Preprocessing- Fit transform train, transform test- return arrays
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Combine features + target into final arrays
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]


            # Save preprocessor
            logging.info('Saving preprocessing object')
            save_object(self.data_transformation_config.preprocessor_obj_file_path,
                        obj= preprocessor)
            
            logging.info("Data transformation completed successfully")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                    )


        except Exception as e:
            raise CustomException(e,sys)
        
        



