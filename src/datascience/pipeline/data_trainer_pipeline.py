from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_trainer import ModelTrainer



class DataTrainerPipeline:
    
    def __init__(self):
        pass 

    def initiate_data_trainer_pipeline(self):
        config=ConfigurationManager()
        data_trainer_config=config.get_model_trainer_config()
        model=ModelTrainer(data_trainer_config)
        model.train()

