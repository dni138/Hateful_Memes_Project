import sentence_encoder
from sklearn.metrics.pairwise import manhattan_distances
class ProtectedClassifierSimple():
    
    def __init__(self, sentence_encoder_model:sentence_encoder.SentenceTransform):
        cls1=["Ethnicity","Religion","Sexual Orientation","Gender","Gender Identity","Disability Disease","Nationality",
      "Immigration Status","socioeconomic Status"]
        ex1=["african","muslim","lesbian","male","transgender","autistic","nationality","migrant","poor"]
        ex2=["white","jewish","gay","woman","queer","cancer","nationality","undocumented","rich"]
        
        self.class_names=cls1
        
        self.transformer=sentence_encoder_model
        
        self.comparison_names=[]
        self.comparison_names.append(ex1)
        self.comparison_names.append(ex2)
        
        self.featurized_cl1=sentence_encoder_model.extract_text_features_simple(cls1)
        #self.featurized_ex1=sentence_encoder_model.extract_text_features_simple(ex1)
        #self.featurized_ex2=sentence_encoder_model.extract_text_features_simple(ex2)
        
    def measure_distance(self, sentences:list):
        
        transformed_sentences:list=self.transformer.extract_text_features_simple(sentences)
            
        #first= manhattan_distances(transformed_sentences,self.featurized_ex1)
        #second=manhattan_distances(transformed_sentences,self.featurized_ex2)
        third=manhattan_distances(transformed_sentences,self.featurized_cl1)
        
        return third #, np.var((first,second,third),axis=0)
        
        
        