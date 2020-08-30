from sentence_transformers import SentenceTransformer
import torch

class SentenceTransform():
    """
     A wrapper for sentence Transformer, maybe will add wrapeprs for ohter stuff, but almost pointless
    """
    
    def __init__(self, name='bert-base-nli-mean-tokens'):
        self.model= SentenceTransformer(name)
        self.name=name

    def extract_text_features_simple(self,data):
        return self.model.encode(data)
    

    
class GettySentenceTransform(SentenceTransform):
    """
    A text transformer with wrappers for our getty dataset that forest made
    """
    
    def __init__(self,name='bert-base-nli-mean-tokens'):
        
        super().__init__(name)
        
    
    def embed_mean(self,word_embedding_model,text_column:list,max_number:int):
        """
        mean embedding of the captions 
        """
        
        assert ("glove" in self.name)==False, "this only runs with None Glove"
        counter=0
        all_captions=[]
        for caption in text_column:
            if counter>max_number:
                break 
            if len(caption)==0:
                continue
            else:
                result=word_embedding_model.extract_text_features_simple([caption])
                all_captions.append(result)
            counter+=1

        mega=torch.Tensor(all_captions)
        return mega.view(mega.shape[0],mega.shape[2]).mean(dim=0)
    
    def embed_column(self,word_embedding_model, text_column:list, mean_embedd, max_number:int):
        """
        same as embed mean, but uses the embed mean to fill empty NAN's in effect
        """
        
        assert ("glove" in self.name)==False, "this only runs with None Glove"
        counter=0
        all_captions=[]
        for caption in text_column:
            if counter>max_number:
                break 
            if len(caption)==0:
                all_captions.append([mean_embedd])

            else:
                result=word_embedding_model.extract_text_features_simple([caption])
                all_captions.append(result)
            counter+=1
        return all_captions
        
        
        
    
    def embed_glove_tags(self,glove_embedding_model,tag_column_list,mean_embed, max_number):
        assert "glove" in self.name, "this only runs with Glove average_word_embeddings_glove.6B.300d"
        
        
        final_data=[]
        for i in range(max_number):
            tags_to_look_at=tag_column_list[i]


            result=self._select_tags_glove(glove_embedding_model,tags_to_look_at)

            result=torch.Tensor(result).mean(dim=0).detach().numpy()
            final_data.append(result)

        return final_data

    def _select_tags_glove(self,glove_embedding_model,tag_string:str):
        #just take out the tags that are actual words 
        tags=tag_string.replace("  "," ").split(" ")

        sub_tags=[]
        for tag in tags: 
            if(tag!=""):
                sub_tags.append(tag)


        temp_result=glove_embedding_model.extract_text_features_simple(sub_tags)
        final_final=[]

        for temp in temp_result:
            if temp[0]!=0:

                final_final.append(temp)

        return final_final


    
    
    
    