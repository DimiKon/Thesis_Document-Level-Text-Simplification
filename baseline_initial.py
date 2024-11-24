# Import necessary libraries

import pandas as pd
import numpy as np
import pickle
import features
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#%%
####### Functions to Parse Datasets #######


def load_simple_target(simple, simple_file):
    '''
    -- gets simple texts from csv dataset and appends it a txt file --
        
       inputs:
            - simple: a simplified text in string format
            - simple_file: output file path                         
    ''' 
    with open(simple_file, 'a', encoding='utf-8') as file:
        simple = simple.replace("\n", " ")
        simple = simple.replace("\r", " ")
        file.write(simple + '\n')
        
        
        
def load_sentences(text_to_sentences, sentences_file):
    '''
    -- gets sentences of a text and appends them to a txt file --
        
       inputs:
            - text_to_sentences: a list of a text split into sentences
            - sentences_file: output file path                         
    '''
    with open(sentences_file, 'a', encoding = 'utf-8') as file:
        for sentence in text_to_sentences:
            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("\r", " ")
            file.write(sentence + '\n')
            
        
def parse_file(file, mode = None):
    '''
    -- parses a csv file to extract features, labels, and keep true simplified texts,
       keep sentences, and lengths of texts in terms of sentences counts, depending on selected mode --
      
     inputs: 
         - file (file path): a csv file
         - mode (str, default = None): defines the process to follow depending on the dataset (train, validation, test)
    '''
    file = pd.read_csv(file)
    all_labels = np.array(file['label'])
    file['sentence'] = file[['text_id', 'source','target','sentence_id', 'sentence', 'label']].groupby(['text_id'])['sentence'].transform(lambda x: '^'.join(x) + '^')
    file['sentence'] = file.sentence.str[0:-1].str.split('^').tolist()
    wo_dup = file.drop_duplicates(subset=['text_id'])
    
    
    feat_1 = []
    feat_2 = []
    feat_3 = []
    feat_4 = []
    feat_5 = []
    feat_6 = []
    feat_7 = []
    feat_8 = []
    feat_9 = []
    feat_10 = []
    feat_11 = []
    
    
    
    num_of_sentences_per_text = []
    for ind in wo_dup.index:
        text_to_sentences = wo_dup['sentence'][ind]
        simplified_text = wo_dup['target'][ind]
        tokenized_sentences = features.preprocessing(text_to_sentences)
        tagged_sentences = features.preprocessing(text_to_sentences, text_type='POS')
        
        #feature extraction
        numerals_scores = features.get_POS_features(tagged_sentences)[1]
        coordinate_clauses_scores = features.get_POS_features(tagged_sentences)[0]
        subordinate_clauses_scores = features.get_POS_features(tagged_sentences)[2]
        flesch = features.readability_metrics(text_to_sentences)[0]
        dale = features.readability_metrics(text_to_sentences)[1]
        difficult_words = features.readability_metrics(text_to_sentences)[2]
        length_scores = features.length_sentence(tokenized_sentences)
        position_scores = features.sentence_position(tokenized_sentences)
        tfisf_scores = features.tfIsf(tokenized_sentences)
        sent_to_sent_similarity = features.sent_to_sent_cohesion(tokenized_sentences)
        sent_to_most_predictive_similarity = features.similarity_predictive_sentence(tokenized_sentences, tfisf_scores)
        
        
        feat_1+= [score for score in numerals_scores]
        feat_2+= [score for score in coordinate_clauses_scores]
        feat_3+= [score for score in subordinate_clauses_scores]
        feat_4+= [score for score in flesch]
        feat_5+= [score for score in dale]
        feat_6+= [score for score in difficult_words]
        feat_7+= [score for score in length_scores]
        feat_8+= [score for score in position_scores]
        feat_9+= [score for score in tfisf_scores]
        feat_10+= [score for score in sent_to_sent_similarity]
        feat_11+=[score for score in sent_to_most_predictive_similarity]
    
  
        if mode == 'test' or mode == 'validation':
            load_simple_target(simplified_text, 'true_simple')
            load_sentences(text_to_sentences, 'sentences')
            num_of_sentences_per_text += [len(text_to_sentences)]
        
     
    feature_matrix = np.stack([(np.array(feat_1)), (np.array(feat_2)), (np.array(feat_3)), (np.array(feat_4)),
                               (np.array(feat_5)), (np.array(feat_6)), (np.array(feat_7)),(np.array(feat_8)),
                               (np.array(feat_9)),(np.array(feat_10)), (np.array(feat_11))], axis=1)    
   
    print(all_labels.shape)
    print(feature_matrix.shape)
    
    #write files
    pickle.dump(feature_matrix, open('features', 'wb'))
    
   
    
    pickle.dump(all_labels, open('labels', 'wb'))
           
    if mode == 'test' or mode == 'validation':
        pickle.dump(num_of_sentences_per_text, open('num_of_sentences_per_text', 'wb'))
        
#%%


####### Training #######


def train(feature_pickle, label_pickle):

    '''
    -- trains a SVM model using grid to test and select the best hyperparameters (C, gamma)
       and writes the best model to a pickle file --
      
     inputs: 
         - feature_pickle (file path): a pickle file with the feature matrix
         - label_pickle (file path): a pickle file with the labels vector
    '''
    X = pickle.load(open(feature_pickle, 'rb'))
    y = pickle.load(open(label_pickle, 'rb'))


    model = SVC(C= 1.5, gamma = 0.1, probability = True, random_state=0)
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', model)])
    model = pipe.fit(X, y)
    pickle.dump(model, open('model', 'wb'))





#%%

####### Predict #######

def predict(feature_pickle, model_pickle, labels_pickle, mode = None):
    '''
    -- implements the trained SVR model on either test or validation dataset to get the predicted score
        per sentence --
      
     inputs: 
         - feature_pickle (file path): a pickle file with the feature matrix
         - model_pickle (file path): a pickle file with the trained model
         - model(str, default = None): defines the process to follow depending on the dataset (validation, test)
     returns:
         - y_pred(list): an array of the predicted labels per text, one label per sentence
    '''
    X_val = pickle.load(open(feature_pickle, 'rb'))
    model = pickle.load(open(model_pickle, 'rb'))
    y_pred = (model.predict_proba(X_val)[:,1] >= 0.564440).astype(bool) 
    #arr0 = [0] * 9919
    #y_pred = arr0
    y_true = pickle.load(open(labels_pickle, 'rb'))
    MSE = log_loss(y_true, model.predict_proba(X_val)[:, 1])
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score (y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names = ['deleted', 'not deleted'], digits=10)
    #matrix = confusion_matrix(y_true, y_pred)
    #sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)
    #plt.xlabel('true label')
    #plt.ylabel('predicted label')
    pred_prob1 = model.predict_proba(X_val)
    fpr1, tpr1, thresh1 = roc_curve(y_true, pred_prob1[:,1], pos_label=1)
    random_probs = [0 for i in range(len(y_true))]
    p_fpr, p_tpr, _ = roc_curve(y_true, random_probs, pos_label=1)
    auc_score1 = roc_auc_score(y_true, pred_prob1[:,1])
    plt.style.use('seaborn')
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='SVC_Baseline')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('ROC',dpi=300)
    plt.show();
    print(auc_score1)
    print(f"Loss:{MSE}")
    print(f"roc_auc_score:{auc_score1}")
    print(f"accuracy:{accuracy}")
    print(f"f1:{f1}")
    print(f"metrics:{report}")
    return y_pred


#%%
# Train Dataset

#x = parse_file('./train/train_data.csv')
#train('./train/features', './train/labels')

#%%
# Validation Dataset
#parse_file('./dev/val_data.csv', mode = 'validation')
#y_pred = predict('./dev/features', './train/model', './dev/labels')


#%%
# Test Dataset
#parse_file('./test/test_data.csv', mode = 'test')
#y_pred = predict('./test/features', './train/model', './test/labels')
#y_pred = predict('./test/features', './train/model', './test/labels')


        
#%%  

                           
    











