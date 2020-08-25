import numpy as np
import pandas as pd
import json

class LoadingData():
    def __init__(self, filepath):
        self.filepath = filepath
        self.filetype = self.filepath.split('.')[-1]
    
    
    def load_data(self):
        data = []
        if 'json' in self.filetype:
            data = []
            with open(self.filepath) as f:
                for el in f:
                    data.append(json.loads(el))
            return data
                    
        else:
            data = pd.read_csv(self.filepath, index_col = False)
            return data
    
