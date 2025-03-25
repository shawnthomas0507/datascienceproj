from src.datascience import logger
import os 
from src.datascience.components.data_transformation import Transformation
from src.datascience.config.configuration import ConfigurationManager

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass 

    def initiate_data_transformation(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation_pipeline=Transformation(data_transformation_config)
        data_transformation_pipeline.transform_data()




