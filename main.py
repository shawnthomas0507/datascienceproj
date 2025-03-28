from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascience.pipeline.data_trainer_pipeline import DataTrainerPipeline
from src.datascience.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

STAGE_NAME="Data Ingestion Stage"

try:
        logger.info(f"stage {STAGE_NAME} started")
        obj=DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        logger.info(f"stage {STAGE_NAME} completed")
    
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME="Data Validation Stage"

try:
        logger.info(f"stage {STAGE_NAME} started")
        obj=DataValidationTrainingPipeline()
        obj.initiate_data_validation()
        logger.info(f"stage {STAGE_NAME} completed")
    
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME="Data Transformation Stage"

try:
        logger.info(f"stage {STAGE_NAME} started")
        obj=DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f"stage {STAGE_NAME} completed")
    
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME="Model Training Stage"

try:
        logger.info(f"stage {STAGE_NAME} started")
        obj=DataTrainerPipeline()
        obj.initiate_data_trainer_pipeline()
        logger.info(f"stage {STAGE_NAME} completed")
    
except Exception as e:
        logger.exception(e)
        raise e







STAGE_NAME="Model Evaluation Stage"

try:
        logger.info(f"stage {STAGE_NAME} started")
        obj=ModelEvaluationPipeline()
        obj.initiate_model_eval_pipeline()
        logger.info(f"stage {STAGE_NAME} completed")
    
except Exception as e:
        logger.exception(e)
        raise e