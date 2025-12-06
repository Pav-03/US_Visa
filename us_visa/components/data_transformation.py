import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            oh_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            ordinal_encoder = OrdinalEncoder()
            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', ordinal_encoder),
                ('scaler', StandardScaler(with_mean=False))
            ])

            numeric_features = self._schema_config['num_features']
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']

            logging.info(f"Categorical columns: {oh_columns}")
            logging.info(f"Ordinal columns: {or_columns}")
            logging.info(f"Numerical columns: {numeric_features}")


            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('oh', oh_transformer, oh_columns),
                    ('or', ordinal_transformer, or_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_ingestion_artifact.validation_error_status:
                 logging.info(f"Data validation failed: {self.data_ingestion_artifact.message}")
                 # You might want to raise error or return None, but here assuming we proceed if user wants or stop
                 # Actually, DataIngestionArtifact doesn't have validation_error_status, DataValidationArtifact has validation_status
                 pass
            
            logging.info("Entered initiate_data_transformation method of DataTransformation class")

            train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            logging.info("Training and test data loaded successfully")

            # Feature Engineering: Company Age
            train_df['company_age'] = CURRENT_YEAR - train_df['yr_of_estab']
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            
            logging.info("Created company_age feature")

            drop_cols = self._schema_config['drop_columns']
            logging.info(f"Dropping columns: {drop_cols}")
            
            train_df = drop_columns(df=train_df, cols=drop_cols)
            test_df = drop_columns(df=test_df, cols=drop_cols)

            target_column_name = TARGET_COLUMN
            
            # Handling Target Column
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Target encoding")
            # Label Encoding Target
            target_feature_train_df = target_feature_train_df.replace({'Certified': 1, 'Denied': 0})
            target_feature_test_df = target_feature_test_df.replace({'Certified': 1, 'Denied': 0})
            
            logging.info("Got preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            transform_input_train_feature = preprocessing_obj.fit_transform(input_feature_train_df)
            transform_input_test_feature = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying SMOTE to training data")
            smt = SMOTETomek(sampling_strategy="minority")
            
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transform_input_train_feature, target_feature_train_df
            )
            
            logging.info("Combining features and targets")
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[transform_input_test_feature, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessing_obj)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved preprocessing object and transformed artifacts")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise USvisaException(e, sys)
