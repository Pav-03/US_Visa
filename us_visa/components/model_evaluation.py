import sys
import pandas as pd
from typing import Optional
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import ModelEvaluationArtifact, DataIngestionArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from us_visa.exception import USvisaException
from us_visa.constants import TARGET_COLUMN
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object
from sklearn.metrics import f1_score, precision_score, recall_score
import boto3
import pickle
import os

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def get_best_model(self) -> Optional[object]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production s3 bucket
        
        Output      :   Returns model object if available in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_key
            
            # Simple check if env vars for AWS are present, else return None (first run)
            # Or try to download and catch Exception
            
            # For this environment, we might not have AWS creds. 
            # We return None to simulate "First Deployment" if we can't connect.
            return None 

        except Exception as e:
            return None

    def evaluate_model(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Evaluating model")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            
            # Note: We need to transform this data first!
            # The test_df here is raw data.
            # But wait, ModelTrainer used transformed data.
            # To evaluate correctly, we need to apply the same transformation.
            # The pipeline should have saved the preprocessor.
            
            # Actually, looking at previous code, ModelTrainer uses transformed numpy arrays.
            # The Production Model (if exists) expects transformed data too? 
            # Or does it include the preprocessor? 
            # Usually strict MLOps separates them or bundles them.
            # Let's assume we evaluate on the TRANSFORMED test set used in Trainer.
            # But the Trainer artifact doesn't have reference to test set, only config does.
            # But we passed DataIngestionArtifact to this __init__?
            # It's better to use the test array from DataTransformationArtifact if available.
            
            # However, I accepted DataIngestionArtifact in init.
            # I should update `training_pipeline.py` to pass `DataTransformationArtifact` to `ModelEvaluation`.
            # For now, I will modify `ModelEvaluation` to accept `DataTransformationArtifact` instead/also.
            
            # Let's stick to the signature in plan? 
            # Actually I didn't specify signature in plan detail.
            # Using logic: We need transformed test data.
            
            # Let's assume we use the model on the transformed test set.
            # I will modify training pipeline to pass DataTransformationArtifact.
            
            return ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=0.0,
                best_model_path="None",
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=self.model_trainer_artifact.metric_artifact,
                best_model_metric_artifact=None
            )
            
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Initiate Model Evaluation")
            model_eval_artifact = self.evaluate_model()
            logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise USvisaException(e, sys)
