#'/opt/conda/bin/python'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import seaborn as sns
import json
import cv2
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils import data as data_torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import os 
import pandas as pd
###
'''
Take all the text features and convert them into Text
Take all the images and featurize them

'''




#https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a THE ANIMAL THIS IS MOST ASSoICATED with = absolutely need to swap out with mask rcnn
def extract_image_features_stupid(data, model, size):
    '''
    THIS NEEDS TO BE DONE IN BATCHES OR YOU'LL RUN OUT OF MEMORY
    Converts. resnet features about. animals for use in our meme model 
    '''
    transform = transforms.Compose(
    [transforms.Resize(size),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    image_transformed=torch.empty(len(data),3,size[0],size[1])
    
    for image_counter in range(len(data)):  
        img_mod=transform(data[image_counter]) 
        image_transformed[image_counter]=img_mod
        image_transformed[image_counter]
    
    return model(image_transformed).detach()
    

    
def extract_text_features_simple(data, model):
    return model.encode(data)


def writeFiles(path:str,title:str,desc:[[]],  img_features:[[]],numpy_form_txts:[[]]):

    results=pd.DataFrame(numpy_form_txts)
    results.to_csv(os.path.join(path,title+"text_features.csv"),sep=",")
    
    results=pd.DataFrame(desc)
    results.to_csv(os.path.join(path,title+"description.csv"),sep=",")
    
    results=pd.DataFrame(img_features)
    results.to_csv(os.path.join(path,title+"image_features.csv"),sep=",")
    
    

def test():
    image_ex_path="/home/data/meme_data/"
    data_ex_path="/home/jupyter/Hateful_Memes_Data/babyData/train.jsonl"
    bert_path="/home/jupyter/ThreeDAnime/bert_folder"
    featurized_data_path="/home/data/meme_challenge_mod_data/"
    
    inception = models.resnet18()



    model_text=SentenceTransformer(bert_path)

    
    data = []
    with open(data_ex_path) as f:
        for el in f:
            data.append(json.loads(el))
            
    
    #labels=[]
    #texts=[]
    desc=[]
    numpy_form_txts=[]
    numpy_form_ims=[]
    for i in range(100):
        print(i)
        try:
            im=Image.open(os.path.join(image_ex_path,data[i]["img"]))
            numpy_form_im=extract_image_features_stupid([im],inception, (299,299))[0].detach().numpy()
            numpy_form_ims.append(numpy_form_im)
            label=data[i]["label"]

            text= data[i]["text"]

            numpy_form_txt=extract_text_features_simple([text],model_text)[0]
            numpy_form_txts.append(numpy_form_txt)
            if(i%10)==0:
                print(i)
            desc.append([text,label])
        except:
            print("failure at"+str(i))


            
    #results=pd.DataFrame(numpy_form_txts)
   # results.to_csv(os.path.join(featurized_data_path,"text.csv"),sep=",")
    
    writeFiles(featurized_data_path, "simple_",desc,numpy_form_ims,numpy_form_txts)
    
if __name__ == "__main__":
    test()
    
    #test(data_ex_path,image_ex_path,) 
    
    
    