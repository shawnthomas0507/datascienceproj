from src.datascience.entity.config_entity import ModelEvaluationConfig
import pandas as pd 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from urllib.parse import urlparse 
import mlflow 
import mlflow.sklearn 
import numpy as np 
import joblib 
from src.datascience.utils.common import save_json
from pathlib import Path
import os 
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/shawnthomas0507/datascienceproj.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="shawnthomas0507"
os.environ['MLFLOW_TRACKING_PASSWORD']="2061b8589608029d512100109903e8a93d0b107d"



class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config=config 

    def eval_metrics(self,actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse,mae,r2
    

    def log_into_mlflow(self):
        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)

        test_x=test_data.drop([self.config.target_column],axis=1)
        test_y=test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_uri_type_store=urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():
            predicted_qualities=model.predict(test_x)

            (rmse,mae,r2_score)=self.eval_metrics(predicted_qualities,test_y)

            scores={"rmse":rmse,"mae":mae,"r2_score":r2_score}
            save_json(path=Path(self.config.metric_file_name),data=scores)

            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("mae",mae)
            mlflow.log_metric("r2_score",r2_score)


            if tracking_uri_type_store!="file":
                mlflow.sklearn.log_model(model,"model",registered_model_name="Elastic model1")
            else:
                mlflow.sklearn.log_model(model,"model")
    