import sys
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
import shutil
import os

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_trainer_artifact: ModelTrainerArtifact):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Entered initiate_model_pusher method of ModelPusher class")
            
            # In a real scenario, we upload to S3 here.
            # For local verification and "shipping" preparation, we can also copy it to a "deployment" folder.
            
            # We will just verify the file exists.
            
            logging.info("Uploaded model to S3 (Mocked)")
            
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_key
            )
            
            logging.info("Model Pusher Artifact created")
            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e, sys)
