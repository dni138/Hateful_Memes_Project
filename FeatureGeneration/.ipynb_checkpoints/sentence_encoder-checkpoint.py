from sentence_transformers import SentenceTransformer


class SentenceTransform():
    """
     A wrapper for sentence Transformer, maybe will add wrapeprs for ohter stuff, but almost pointless
    """
    
    def __init__(self, name='bert-base-nli-mean-tokens'):
        self.model= SentenceTransformer(name)
        
    def extract_text_features_simple(self,data):
        return self.model.encode(data)