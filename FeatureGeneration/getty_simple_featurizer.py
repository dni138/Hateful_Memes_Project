import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk import word_tokenize

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from LoadingData import LoadingData

class GettySimpleWrapper():
    
    def __init__(self):
        self.data = LoadingData.LoadingData("~/projects/Hateful_Memes_Project/data/getty_data_full.csv")
    
    
    def clean_getty_data(self):
        getty=self.data.load_data()
        df=getty[~getty['image'].str.contains("\(")]
        getty[~getty['image'].str.contains("\(")].image.nunique()

        df['color_zscore'] = (df['Color_Score'] - df['Color_Score'].mean())/df['Color_Score'].std(ddof=0)
        df['key_point_zscore'] = (df['Key_Point_Score'] - df['Key_Point_Score'].mean())/df['Key_Point_Score'].std(ddof=0)
        df['combined_z']=df['color_zscore']+df['key_point_zscore']
        df['combined_zscore']=(df['combined_z'] - df['combined_z'].mean())/df['combined_z'].std(ddof=0)

    #     s = df.combined_zscore
    #     ax = s.plot.kde()
    #     display(df.combined_zscore.describe())

        combined_index = df.groupby('image')['combined_zscore'].idxmin()
        match=df.loc[combined_index]

        return match
    
    
    def load_full_dataset(self):
        df_list = []

        for x in ['train','dev','test']:
            file="../data/"+x+".jsonl"
            data=LoadingData.LoadingData(file)
            data_json=data.load_data()
            df=pd.DataFrame(data_json)
            df['partition']=x
            if x=='test':
                df['label']=np.nan
            df_list.append(df)

        final = pd.concat(df_list)
        return final

    
    def create_simple_getty_features(self):
        #load data.  Match is best matching Getty data and df is full dataframe of train/test/dev data
        match=self.clean_getty_data()
        df=self.load_full_dataset()


        #clean up types, column names, and text
        match['src']=match['src'].astype(str)
        df['id']=df['id'].astype(str)
        match.rename(columns={'id':'rank'}, inplace=True)

        full=df.merge(match, how='left', left_on='id', right_on='src')
        full['tags_clean']= full['tags'].str.replace(',',' ').str.replace('Photos',' ').str.lower().str.split()

        #tokenize text
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        full['text_clean'] = full['text'].apply(lambda x: tokenizer.tokenize(x))
        
        clean=full[['id','img','text','text_clean','partition','tags','tags_clean','caption','combined_zscore','label']]

        #calculate text frequencies
        clean['all_text_freq']= clean.text.map(clean.text.value_counts())
        clean['train_text_freq']= clean.text.map(clean[clean.partition=='train'].text.value_counts())
        clean['dev_text_freq']= clean.text.map(clean[clean.partition=='dev'].text.value_counts())
        clean['test_text_freq']= clean.text.map(clean[clean.partition=='test'].text.value_counts())

        # manual lists for features.  This snowballed into something ugly, refactor in a better way, maybe dict.
        race=['african','black','asian','chinese','arab','white','african-american','ethnicity']
        disability=['disabled','disability',"down's, syndrome",'retarded','retarted']
        religion=['islam','muslim','muslims','catholic','catholics','christian','christians','jewish','jew','jews',
                 'god','jesus']
        sexual_orientation=['gay','straight','trans','transexual','homo','homosexual']
        violence=['obscene','anger','aggression','kill','killing','bomb','gun']

        criminals=['hitler']
        animals=['alligator','ant','bear','bee','bird','camel','cat','cheetah','chicken','chimpanzee','cow','crocodile','deer','dog','dolphin','duck','eagle','elephant','fish','fly','fox','frog','giraffe','goat','goldfish','hamster','hippopotamus','horse','kangaroo','kitten','lion','lobster','monkey','octopus','owl','panda','pig','puppy','rabbit','rat','scorpion','seal','shark','sheep','snail','snake','spider','squirrel','tiger','turtle','wolf','zebra']

        gender=['man','woman','men','women','bitch','pussy']

        #simple features for baseline
        clean['tags_clean'] = [ [] if x is np.NaN else x for x in clean['tags_clean'] ]
        clean['tags_race'] = clean.apply(lambda row: True if any(item in race for item in row['tags_clean']) else False, axis = 1)
        clean['tags_disability'] = clean.apply(lambda row: True if any(item in disability for item in row['tags_clean']) else False, axis = 1)
        clean['tags_religion'] = clean.apply(lambda row: True if any(item in religion for item in row['tags_clean']) else False, axis = 1)
        clean['tags_sexual_orientation'] = clean.apply(lambda row: True if any(item in sexual_orientation for item in row['tags_clean']) else False, axis = 1)
        clean['tags_violence'] = clean.apply(lambda row: True if any(item in violence for item in row['tags_clean']) else False, axis = 1)
        clean['tags_criminals'] = clean.apply(lambda row: True if any(item in criminals for item in row['tags_clean']) else False, axis = 1)
        clean['tags_gender'] = clean.apply(lambda row: True if any(item in gender for item in row['tags_clean']) else False, axis = 1)
        clean['tags_animals'] = clean.apply(lambda row: True if any(item in animals for item in row['tags_clean']) else False, axis = 1)

        clean['text_clean'] = [ [] if x is np.NaN else x for x in clean['text_clean'] ]
        clean['text_race'] = clean.apply(lambda row: True if any(item in race for item in row['text_clean']) else False, axis = 1)
        clean['text_disability'] = clean.apply(lambda row: True if any(item in disability for item in row['text_clean']) else False, axis = 1)
        clean['text_religion'] = clean.apply(lambda row: True if any(item in religion for item in row['text_clean']) else False, axis = 1)
        clean['text_sexual_orientation'] = clean.apply(lambda row: True if any(item in sexual_orientation for item in row['text_clean']) else False, axis = 1)
        clean['text_violence'] = clean.apply(lambda row: True if any(item in violence for item in row['text_clean']) else False, axis = 1)
        clean['text_criminals'] = clean.apply(lambda row: True if any(item in criminals for item in row['text_clean']) else False, axis = 1)
        clean['text_gender'] = clean.apply(lambda row: True if any(item in gender for item in row['text_clean']) else False, axis = 1)
        clean['text_animals'] = clean.apply(lambda row: True if any(item in animals for item in row['text_clean']) else False, axis = 1)

        return clean
    
    
    def get_simple_getty_features(self):
        clean=self.create_simple_getty_features()
        features=['id','all_text_freq','train_text_freq', 'dev_text_freq', 'test_text_freq', 'tags_race',
                  'tags_disability', 'tags_religion', 'tags_sexual_orientation','tags_violence',
                  'tags_criminals', 'tags_gender', 'tags_animals','text_race', 'text_disability',
                  'text_religion','text_sexual_orientation', 'text_violence', 'text_criminals',
                  'text_gender', 'text_animals']
        results=clean[features]

        return results