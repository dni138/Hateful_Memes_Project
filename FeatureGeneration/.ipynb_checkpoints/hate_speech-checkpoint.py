

from hatesonar import Sonar
import pandas as pd

class HateWrapper():
    
    def __init__(self):
        self.sonar=Sonar()
        
    
    def predict(self,texts:list)->pd.DataFrame:
        hate_speech,offensive_lang,neither=[],[],[]
        sonar=self.sonar
        for text in texts:
            row=[]
            hate_speech.append(sonar.ping(text=str(text or '')).get('classes')[0].get('confidence'))
            offensive_lang.append(sonar.ping(text=str(text or '')).get('classes')[1].get('confidence'))
            neither.append(sonar.ping(text=str(text or '')).get('classes')[2].get('confidence'))
            
        results=pd.DataFrame()
        results["text"]=texts
        results["hate_speech"]=hate_speech
        results["offensive_language"]=offensive_lang
        results["neither"]=neither
        
        return results