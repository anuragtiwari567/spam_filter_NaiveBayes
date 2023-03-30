# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:50:32 2018

@author: HRITHIK
"""
#        Data Preprocessing
import pandas as pd
import indicoio
from profanity import profanity
indicoio.config.api_key = '473582e80e2d070f5d496b14f8f83ee8'
data=pd.read_table(r"SMSCollection.txt",header=None,names=['label','message'])
print(data.label.value_counts())
data['label_num']=data.label.map({'ham':0,'spam':1})
X=data.message
y=data.label_num.astype(int)

#        Cross Validation
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

#        Vectorising our dataset
from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(min_df=5,max_df=0.8,use_idf=True,stop_words='english',sublinear_tf=True)
X_train_dtm=vect.fit_transform(X_train)
X_test_dtm=vect.transform(X_test)

#        Bayesian Modelling for text Learning
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_dtm,y_train)
y_test_pred=model.predict(X_test_dtm)

#        Evaluation for Hyperparameters
from sklearn import metrics
print("Null Accuracy(Biased Accuracy leaning towards Majority)= ",y_test.value_counts().head(1)/len(y_test))
print("Accuracy Score= ",metrics.accuracy_score(y_test,y_test_pred))
confusion=metrics.confusion_matrix(y_test,y_test_pred)
print(confusion)
prob=model.predict_proba(X_test_dtm)[:,1]
fpr,tpr,thresholds=metrics.roc_curve(y_test,prob)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.xlim(0,0.4)
plt.ylim(0.9,1)
plt.title("ROC curve for spam filter")
plt.xlabel("fpr=1-Specificity")
plt.ylabel("tpr=specitivity")
plt.grid("True")
print("ROC_AUC_SCORE= ",metrics.roc_auc_score(y_test,prob))
#        Sentiment Analyzer
#Offline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
vectsa=TfidfVectorizer(min_df=5,max_df=0.8,sublinear_tf=True,use_idf=True)
neg_data=pd.read_csv(r"E:\OLD_LAPI\datasets\rt-polaritydata\rt-polaritydata\rt-polarity.neg","negative",names=['words','sentiment'])
neg_data['sentiment']='0'
pos_data=pd.read_csv(r"E:\OLD_LAPI\datasets\rt-polaritydata\rt-polaritydata\rt-polarity.pos","positive",names=['words','sentiment'])
pos_data['sentiment']='1'
data=pd.concat([neg_data,pos_data],axis=0)
from sklearn.naive_bayes import MultinomialNB
data_dtm=vectsa.fit_transform(data.words)
data.sentiment=data.sentiment.astype('int')
clf=MultinomialNB()
clf.fit(data_dtm,data.sentiment)
# =============================================================================
# EXAMINING THE WORKING OF MODEL
# =============================================================================
# =============================================================================
X_train_tokens=vect.get_feature_names()
print(len(X_train_tokens)," Tokens are der in the dtm matrix")
X_train_tokens=vect.get_feature_names()
print(len(X_train_tokens))#no. of tokens
#in naive bayes each token appears in each class
print("Counts each token appeared in each class",model.feature_count_,"Its shape is=",model.feature_count_.shape)
#conditional prob of every token is evaluated for each class.learns spaminess of each token with the hamminess..tht is i it has a lot of spammy wrds its a spam 
#no of times each token appears accross all
ham_token_count=model.feature_count_[0,:]
spam_token_count=model.feature_count_[1,:]#appearance of each token accross all spam msgs
tokens=pd.DataFrame({'token':X_train_tokens,'ham':ham_token_count,'spam':spam_token_count})
print(tokens.head(10))
print("5 random picks",tokens.sample(5,random_state=6))
print("The data I was trained contained (ham,spam===)", model.class_count_)
# =============================================================================
# In order to avoid divide by zero error we add(A hack for the working of naive bayes) 
# =============================================================================
print("Working of naive bayes")
tokens['ham']=tokens.ham+1
tokens['spam']=tokens.spam+1
#Convert the ham and spam counts into frequency
tokens['ham']=tokens.ham/model.class_count_[0]
tokens['spam']=tokens.spam/model.class_count_[1]
print("Ham and spam tokens occurung frequency")
print(tokens.sample(5,random_state=6))
tokens['Spam_ratio']=tokens.spam/tokens.ham
print(tokens.sample(5,random_state=6))
#greater the ratio greater spaminess
print(tokens.sort_values('Spam_ratio',ascending=False))
#tokens.loc('textoperator','Spam_ratio')
#above command is for lookup for a particular word spaminess...
#Tester
from textblob import TextBlob
def spam_filter(msg=input("Enter message = ")):
    msg=TextBlob(msg)
    current_lang=msg.detect_language()
    print("Language of this message is = ",current_lang)
    if (current_lang!='en'):
        msg.translate(to='en')
    else:
        msg.correct()
    X_dtm=vect.fit_transform(X)
    test_dtm=vect.transform([str(msg)])
    model.fit(X_dtm,y)
    result=model.predict(test_dtm)
    prob=model.predict_proba(test_dtm)
    if result==[1]:
        print("SPAM ALERT!")
    else:
        print("HAM")
        predsa=clf.predict(vectsa.transform([str(msg)]))
        
        if predsa==[1]:
             print("Positive Feeling")
             
        elif predsa==[0]:
             print("Negative Feeling")
        else:print("Can't analyze ur Felling...Try API ? ....")
        senti=indicoio.sentiment_hq(str(msg))
        print("Online Help , Positivity of Incoming Message = ",senti)
    p=indicoio.personality(str(msg))
    d=[]
    d.append([p['agreeableness'],p['conscientiousness'],p['extraversion'],p['openness'],msg.sentiment.polarity,msg.sentiment.subjectivity])
    traits=pd.DataFrame(d,columns=['agreeableness','conscientiousness','extraversion','openness','polarity','subjectivity'])
    print(profanity.contains_profanity(str(msg))," Profanity")
    print(profanity.censor(str(msg))) 
    print("Summarizing this message =",msg.noun_phrases)
    percent=pd.DataFrame(prob,columns=["% HAM","%SPAM"])
    print(traits)
    print(percent)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













