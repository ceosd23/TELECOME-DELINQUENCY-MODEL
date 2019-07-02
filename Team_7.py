filename="Micro_delinquency_train_test_dataset_v2.csv" 
#The testing dataset is by default set to trsining datset, to test on different
#dataset change the filename with url to test data.
# =============================================================================
# We should always target to increase sensitivity over specificity as sensitivity
# is the true positive rate whereas specificity is the false positive rate.
#True Positive : These are the correctly predicted positive values which means 
#that the value of actual class is yes and the value of predicted class is also 
#yes.
#False Positive : These are the wrongly predicted positive values which means
# that the value of actual class is no and the predicted class is yes.
#All the important parameters such as Accuracy, Precision and Recall depend upon
# our True positive rate.So if there is a increase in false positive rate then 
#therewill be a downgrade in these parameters which in return will give not so good outcomes.
# =============================================================================
# -*- coding: utf-8 -*-

"""
Created on Saturday Jun 8 10:54:49 2019
by Soumyadeep Ghosh

#############STEP:- IMPORT NECESSARY LIBRARIES##############################
=============================================================================
ALL REQUIRED LIBRARIES ARE AT FIRST LOADED
PANDAS USED FOR THE DATA HANDLING
MATPLOTLIB AND SEABORN USED FOR DATA VISUALIZATION
Numpy Provides fater data handling and also important mathematical features
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


"""
#######################STEP:-READ IN AND EXPLORE THE DATA###################
=============================================================================
LOADING OF THE DATA FROM CSV FORMAT FOR BOTH TRAINING OUR MODEL AND TESTING 
PREDICTION.
=============================================================================
"""
data=pd.read_csv('Micro_delinquency_train_test_dataset_v2.csv')
test=pd.read_csv(filename)

#print(data.describe(include='all'))
"""
OUTPUT:-
 	Unnamed: 0 	label 	msisdn 	aon 	daily_decr30 	daily_decr90 	rental30 	rental90 	last_rech_date_ma 	last_rech_date_da 	... 	maxamnt_loans30 	medianamnt_loans30 	cnt_loans90 	amnt_loans90 	maxamnt_loans90 	medianamnt_loans90 	payback30 	payback90 	pcircle 	pdate
count 	190001.000000 	190001.000000 	190001 	190001 	190001 	190001 	190001 	190001 	190001.000000 	190001.000000 	... 	190001.000000 	190001.000000 	190001.000000 	190001.000000 	190001.000000 	190001.000000 	190001.000000 	190001.000000 	190001 	190001
unique 	NaN 	NaN 	170612 	4314 	134590 	142616 	121357 	128203 	NaN 	NaN 	... 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	1 	82
top 	NaN 	NaN 	47819I90840 	95 	0 	0 	0 	0 	NaN 	NaN 	... 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	UPW 	7/4/2016
freq 	NaN 	NaN 	7 	357 	3741 	3669 	6802 	6227 	NaN 	NaN 	... 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	190001 	2857
mean 	95001.000000 	0.875069 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	3735.346070 	3731.710966 	... 	276.634790 	0.054160 	18.364699 	23.639181 	6.701870 	0.046289 	3.401179 	4.322936 	NaN 	NaN
std 	54848.708586 	0.330641 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	53814.346634 	53558.030669 	... 	4267.364414 	0.218396 	223.312555 	26.503844 	2.102752 	0.201153 	8.836749 	10.318272 	NaN 	NaN
min 	1.000000 	0.000000 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	-29.000000 	-29.000000 	... 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	NaN 	NaN
25% 	47501.000000 	1.000000 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	1.000000 	0.000000 	... 	6.000000 	0.000000 	1.000000 	6.000000 	6.000000 	0.000000 	0.000000 	0.000000 	NaN 	NaN
50% 	95001.000000 	1.000000 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	3.000000 	0.000000 	... 	6.000000 	0.000000 	2.000000 	12.000000 	6.000000 	0.000000 	0.000000 	1.666667 	NaN 	NaN
75% 	142501.000000 	1.000000 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	7.000000 	0.000000 	... 	6.000000 	0.000000 	5.000000 	30.000000 	6.000000 	0.000000 	3.750000 	4.500000 	NaN 	NaN
max 	190001.000000 	1.000000 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	997717.809600 	999171.809400 	... 	99864.560860 	3.000000 	4997.517944 	438.000000 	12.000000 	3.000000 	171.500000 	171.500000 	NaN 	NaN

11 rows × 37 columns
"""


######################STEP:- DATA ANALYSIS#########################
# =============================================================================
# 
# =============================================================================

#print(data.shape)
#It help us to undesrtand how much instances we have, which is the row number
#and the columns Number tell us about the feature in the data
"""
(190001, 36)
"""
#print(data.isnull().sum())
#This tells about if the number of rows are null or not
"""
label                   0
msisdn                  0
aon                     0
daily_decr30            0
daily_decr90            0
rental30                0
rental90                0
last_rech_date_ma       0
last_rech_date_da       0
last_rech_amt_ma        0
cnt_ma_rech30           0
fr_ma_rech30            0
sumamnt_ma_rech30       0
medianamnt_ma_rech30    0
medianmarechprebal30    0
cnt_ma_rech90           0
fr_ma_rech90            0
sumamnt_ma_rech90       0
medianamnt_ma_rech90    0
medianmarechprebal90    0
cnt_da_rech30           0
fr_da_rech30            0
cnt_da_rech90           0
fr_da_rech90            0
cnt_loans30             0
amnt_loans30            0
maxamnt_loans30         0
medianamnt_loans30      0
cnt_loans90             0
amnt_loans90            0
maxamnt_loans90         0
medianamnt_loans90      0
payback30               0
payback90               0
pcircle                 0
pdate                   0
dtype: int64
"""

#print(data.dtypes) 
#It Tells us about the data Types of all the feature in the data

"""
Unnamed: 0         Micro_delinquency_train_test_dataset_v2.csv       int64
label                     int64
msisdn                   object
aon                      object
daily_decr30             object
daily_decr90             object
rental30                 object
rental90                 object
last_rech_date_ma       float64
last_rech_date_da       float64
last_rech_amt_ma          int64
cnt_ma_rech30             int64
fr_ma_rech30            float64
sumamnt_ma_rech30       float64
medianamnt_ma_rech30    float64
medianmarechprebal30    float64
cnt_ma_rech90             int64
fr_ma_rech90              int64
sumamnt_ma_rech90         int64
medianamnt_ma_rech90    float64
medianmarechprebal90    float64
cnt_da_rech30           float64
fr_da_rech30            float64
cnt_da_rech90             int64
fr_da_rech90              int64
cnt_loans30               int64
amnt_loans30              int64
maxamnt_loans30         float64
medianamnt_loans30      float64
cnt_loans90             float64
amnt_loans90              int64
maxamnt_loans90           int64
medianamnt_loans90      float64
payback30               float64
payback90               float64
pcircle                  object
pdate                    object
dtype: object
"""


##############################STEP:-Visulazation DATA#############################
# =============================================================================
# We have removed the visulaization of the data because it would increase the 
#complexity and runtime of the program but we would like to draw attention that
#most of the data was having a kind of normalized graph with leptokertic type of graph.
#On plotting of the attribute we found that we were getting distributing with mostly
#one mode.
#data.aon.plot(kind='density')
#plt.show()
#correlations = data.corr()
#print(correlations)
# plot correlation matrix
#fig = plt.figure()
#Following will add matrix and side bar in entire area
#subFig = fig.add_subplot(111)
#cax = subFig.matshow(correlations, vmin=-1, vmax=1)
#fig.colorbar(cax)
#plt.show()
#-----------------------------
#ticks = nd.arange(0,32)   
#subFig.set_xticks(ticks)
#subFig.set_yticks(ticks)
#subFig.set_xticklabels(data.columns)
#subFig.set_yticklabels(data.columns)
#data.hist()
#plt.show()
#sn.barplot(x='aon',y='label',data=data)
#plt.show()
##############################################################################





##############################STEP:-CLEANING DATA#############################
# =============================================================================
#In Machine Learning We can directly work only with numerical values
#On analysing the Various features we found that the features with object type 
#are mostly numeric value but contains misssing values marked as UA So we decid
#-ed to remove it but before that we will extract the label from the data


y_train=data.label
y_test=test.label



#Changing each data into numeric for for both test and train
for i in data.columns:
    if i=='pdate':
        continue
    else:
        data[i]=pd.to_numeric(data[i],errors='coerce')
for i in test.columns:
    if i=='pdate':
        continue
    else:
        test[i]=pd.to_numeric(test[i],errors='coerce')




#Dropping the pcircle as it contains one value in all the rows so it won't help 
#in neither training nor test and also the values in PDATE were removed as it 
#wasn't mentioned what it contains so we removed it
#We also removed the msidn as the cellphone number isn't the deciding factor for
#prediction the deliquency though its necessary for knowing which data is linked
#to which User. 

        
data.drop(['Unnamed: 0','msisdn','pcircle','pdate'],axis=1,inplace=True)
test.drop(['Unnamed: 0','msisdn','pcircle','pdate'],axis=1,inplace=True)


#On changing the value to Numeric all the values containing str was filled with NaN
#so we replaced it with the mode values of that attribute and if the test data also
#contained Missing values then it will be replace by the median of the training data
#because median isn't affected my outliers. 


for i in data.columns:
    data[i].fillna(data[i].median(),inplace=True)
for i in test.columns:
    test[i].fillna(data[i].median(),inplace=True)




#We used the normalizer to stop the spread of the data
#We Normalized both training and testing data 

from sklearn.preprocessing import Normalizer
Normal=Normalizer()
names=[str(i) for i in data.columns]
Normaldata=Normal.fit_transform(data)
data=pd.DataFrame(Normaldata,columns=names)
test.drop(['label'],axis=1,inplace=True)
testnames=[str(i) for i in test.columns]
Normaldata=Normal.fit_transform(test)
test=pd.DataFrame(Normaldata,columns=testnames)
data.drop('label',axis=1,inplace=True)


#We got confused by  the correlation matrix and so dropped the idea of dropping feature
#as we weren't sure so we used RFE for selecting 25 Features and we choosed the 
#RandomForestClassifier as it all ready uses ginni index for selection of feature

x=data.values
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
RF=RandomForestClassifier()
rfe=RFE(RF,25)
fit=rfe.fit(x,y_train)
j=1
name=[]
for i in fit.support_:
    if i==True:
        name.append(names[j])
    j+=1
data_featureSelected=pd.DataFrame()
test_featureSelected=pd.DataFrame()
for i in name:
    data_featureSelected[i]=data[i]
    test_featureSelected[i]=test[i]
data=data_featureSelected.copy()
test=test_featureSelected.copy()

#After Feature Selection We once again used the idea of Zcore and any zcore 
#greater than >3 were remove from the data as 6 Sigma already covers most of the
#population

from scipy import stats
data[(np.abs(stats.zscore(data))<3).all(axis=1)]
test[(np.abs(stats.zscore(test))<3).all(axis=1)]

'''
###############################################################################
===============================================================================
We found that the class of 0 was highly Imbalanced the minority class was about
12.00% and thats why most of the prediction model was predicting higher values 
and to handel this situation we used the concept of Up sampling the data by 
putting duplicate values of Minority
Class
===============================================================================
###############################################################################
'''

from sklearn.utils import resample
data['label']=y_train[:]

df_majority=data[data.label==1]
df_minority=data[data.label==0]
df_upsamp=resample(df_minority,replace=True,n_samples=166264,random_state=7)
df_up=pd.concat([df_majority,df_upsamp])
y_train=df_up.label
df_up.drop('label',axis=1,inplace=True)
data=df_up.copy()

############################STEP :TRAINING THE MODEL##########################
# =============================================================================

model=RandomForestClassifier()
model.fit(data,y_train)

###########################STEP :TESTING THE MODEL ON GIVEN DATA##############
# =============================================================================

prediction=model.predict(test)

##########################STEP: GENERATING THE SCORE FOR MODEL################
# =============================================================================
from sklearn.metrics import accuracy_score,precision_score,recall_score
accuracyscore=round(accuracy_score(prediction,y_test)*100,2)
precScore=round(precision_score(prediction,y_test)*100,2)
recallScore=round(recall_score(prediction,y_test)*100,2)
print("          #####################")
print("          #Accuracy  : ",accuracyscore,'#')
print("          #Precision : ",precScore,'#')
print("          #Recall    : ",recallScore,'#')
print("          #####################")      
