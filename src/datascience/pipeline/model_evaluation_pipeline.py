from src.datascience import logger
from src.datascience.components.model_evaluation import ModelEvaluation
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.utils.common import read_yaml,create_directories,save_json





class ModelEvaluationPipeline:

    def __init__(self):
       pass


    def initiate_model_eval_pipeline(self):
        config=ConfigurationManager()
        model_eval_config=config.get_model_eval_config()
        eval=ModelEvaluation(model_eval_config)
        eval.log_into_mlflow()

