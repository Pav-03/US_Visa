import os
import sys
from datetime import date

import pandas as pd
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object

class USVisaData:
    def __init__(self,
                 continent: str,
                 education_of_employee: str,
                 has_job_experience: str,
                 requires_job_training: str,
                 no_of_employees: int,
                 region_of_employment: str,
                 prevailing_wage: int,
                 unit_of_wage: str,
                 full_time_position: str,
                 yr_of_estab: int
                 ):
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.yr_of_estab = yr_of_estab
            
            # Calculate company_age
            self.company_age = date.today().year - self.yr_of_estab

        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age] 
            }
            # Note: We return dataframe with 'company_age' because our 'DataTransformation' 
            # pipeline expects 'company_age' to exist (it was created before transformation step in training).
            # Wait, in DataTransformation.initiate... I created 'company_age' myself from 'yr_of_estab'.
            # BUT, the preprocessing object (ColumnTransformer) expects 'company_age' as a column in 'num_features'?
            # Let's check schema.yaml:
            # num_features: [no_of_employees, prevailing_wage, company_age]
            # YES. The pipeline (preprocessor) expects 'company_age'.
            # So passing a dataframe with 'company_age' to the preprocessor.transform() is correct.

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise USvisaException(e, sys) from e


class USVisaClassifier:
    def __init__(self, prediction_pipeline_config: dict = None):
        """
        :param prediction_pipeline_config: Configuration for prediction pipeline
        """
        self.model_dir = "artifact/model_trainer/trained_model" 
        self.preprocessor_dir = "artifact/data_transformation/transformed_object"
        
        # In production this would come from the pushed S3 location or a "saved_models" dir
        # For now, pointing to latest artifact location
        
        self.model_path = os.path.join(self.model_dir, "model.pkl")
        self.preprocessor_path = os.path.join(self.preprocessor_dir, "preprocessing.pkl")

    def predict(self, dataframe: pd.DataFrame) -> str:
        try:
            logging.info("Entered predict method of USVisaClassifier class")
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            logging.info("Transofrming input data")
            # The preprocessor pipeline handles scaling/encoding.
            transformed_data = preprocessor.transform(dataframe)
            
            logging.info("Predicting")
            prediction = model.predict(transformed_data)
            
            return prediction
        
        except Exception as e:
            raise USvisaException(e, sys)
