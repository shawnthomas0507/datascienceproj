import urllib.request as request
from src.datascience import logger
import zipfile
import os
from src.datascience.entity.config_entity import (DataIngestionconfig)

import requests
class DataIngestion:
    def __init__(self,config:DataIngestionconfig):
        self.config=config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename,headers=request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with info {headers}")
        else:
            logger.info("file already exists")

    
    def extract_zip(self):
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        

