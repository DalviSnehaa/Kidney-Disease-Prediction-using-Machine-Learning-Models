#!/usr/bin/env python
# coding: utf-8

# ## Kidney Disease Prediction Model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set()
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


kd = pd.read_csv(r"C:\datasets\kidney_disease.csv")


# In[3]:


kd


# In[4]:


kd.head()


# In[5]:


kd.shape


# In[6]:


kd.classification.value_counts()


# In[7]:


kd.shape


# In[8]:


kd = kd.drop(['id'] ,axis= 1)


# In[9]:


kd.head(2)


# In[10]:


kd.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']


# In[11]:


kd.head()


# In[12]:


kd.age.describe()


# In[13]:


kd.info()


# In[14]:


kd['packed_cell_volume'] = pd.to_numeric(kd['packed_cell_volume'], errors='coerce')
kd['white_blood_cell_count'] = pd.to_numeric(kd['white_blood_cell_count'], errors='coerce')
kd['red_blood_cell_count'] = pd.to_numeric(kd['red_blood_cell_count'], errors='coerce')


# In[15]:


kd.head()


# In[16]:


kd.groupby('red_blood_cells').red_blood_cells.count()


# In[17]:


cat_cols = [col for col in kd.columns if kd[col].dtype == 'object']
num_cols = [col for col in kd.columns if kd[col].dtype != 'object']


# In[18]:


cat_cols


# In[19]:


num_cols


# In[20]:


kd.isnull().sum()[kd.isnull().sum() > 0]


# In[21]:


kd.age.fillna(kd.age.mean(), inplace= True)
kd.blood_pressure.fillna(kd.blood_pressure.mean(), inplace= True)
kd.specific_gravity.fillna(kd.specific_gravity.mean(), inplace= True)
kd.albumin.fillna(0.0 , inplace= True)
kd.sugar.fillna(0.0 , inplace= True)
kd.red_blood_cells.fillna('abnormal', inplace=True)
kd.pus_cell.fillna('abnormal', inplace=True)
kd.pus_cell_clumps.fillna('present' , inplace=True)
kd.bacteria.fillna('present' , inplace=True)
kd.blood_glucose_random.fillna(kd.blood_glucose_random.mean(), inplace= True)
kd.blood_urea.fillna(kd.blood_urea.mean(), inplace= True)
kd.serum_creatinine.fillna(kd.serum_creatinine.mean(), inplace= True)
kd.sodium.fillna(kd.sodium.mean(), inplace= True)
kd.potassium.fillna(kd.potassium.mean(), inplace= True)
kd.haemoglobin.fillna(kd.haemoglobin.mean(), inplace= True)
kd.packed_cell_volume.fillna(kd.packed_cell_volume.mean(), inplace= True)
kd.white_blood_cell_count.fillna(kd.white_blood_cell_count.mean(), inplace= True)
kd.red_blood_cell_count.fillna(kd.red_blood_cell_count.mean(), inplace= True)
kd.hypertension.fillna('yes', inplace=True)
kd.diabetes_mellitus.fillna('yes', inplace=True)
kd.coronary_artery_disease.fillna('yes', inplace=True)
kd.appetite.fillna('poor', inplace=True)
kd.peda_edema.fillna('yes', inplace=True)
kd.aanemia.fillna('yes', inplace= True)


# In[22]:


kd.diabetes_mellitus.replace({'yes':'yes', 'no':'no', ' yes':'yes', '\tno':'no', '\tyes':'yes'}, inplace=True)
kd.coronary_artery_disease.replace({'no':'no', 'yes':'yes','\tno':'no' }, inplace=True)
kd['class'] = kd['class'].replace(to_replace={'ckd\t':'ckd', 'notckd': 'not ckd'})


# In[23]:


kd['class'].value_counts()


# In[ ]:





# In[24]:


kd.pus_cell.value_counts()


# In[25]:


kd['class'].unique()


# In[26]:


kd.red_blood_cells.replace({'abnormal':0, 'normal':1}, inplace= True)
kd.pus_cell.replace({'abnormal':0, 'normal':1}, inplace= True)
kd.pus_cell_clumps.replace({'notpresent':0, 'present':1}, inplace=True)
kd.bacteria.replace({'notpresent':0, 'present':1}, inplace=True)
kd.hypertension.replace({'yes':1, 'no':0}, inplace= True)
kd.diabetes_mellitus.replace({'yes':1, 'no':0}, inplace= True)
kd.coronary_artery_disease.replace({'yes':1, 'no':0}, inplace= True)
kd.appetite.replace({'good':1, 'poor':0},inplace=True)
kd.peda_edema.replace({'yes':1, 'no':0}, inplace= True)
kd.aanemia.replace({'yes':1, 'no':0}, inplace= True)
kd['class'].replace({'ckd':1, 'not ckd':0}, inplace= True)


# In[27]:


kd.info()


# In[28]:


print(kd.groupby('class').red_blood_cell_count.mean().plot(kind="bar", color='teal'))


# In[29]:


# Counting number of normal vs. abnormal red blood cells of people having chronic kidney disease
print(kd.groupby('red_blood_cells').red_blood_cells.count().plot(kind="bar", color='green'))


# In[30]:


# Counting number of normal vs. abnormal pus cells of people having chronic kidney disease
print(kd.groupby('pus_cell').pus_cell.count().plot(kind="bar"))


# In[31]:


print(kd.groupby('hypertension').age.mean().plot(kind='bar', color='green'))


# In[32]:


#This plot shows the patient's sugar level compared to their ages

kd.plot(kind='scatter', x='age',y='sugar', color ='green');
plt.show()


# In[33]:


## Shows the maximum blood pressure having chronic kidney disease
# 0 : having chronic disease
#1 : not having Chronic disease
print(kd.groupby('class').blood_pressure.max().plot(kind="bar", color = 'red'))


# In[34]:


num_cols1 = num_cols[:-2]
fig = px.box(kd[num_cols1], y=num_cols1)
fig.show()


# In[35]:


plt.hist(kd.age , color= 'green', bins = 40 , edgecolor = "yellow");
# seems normally distibuted


# In[36]:


plt.hist(kd.white_blood_cell_count , color= 'red', bins = 40 , edgecolor = "yellow");


# In[37]:


plt.hist(kd.blood_glucose_random , color= 'teal', bins = 40 , edgecolor = "yellow");


# In[38]:


# average age of people having ckd and not having ckd
# 0 : not having ckd
# 1 : having ckd

print(kd.groupby('class').age.mean().plot(kind="bar", color='green'))


# In[39]:


plt.figure(figsize = (20, 15))
plotnumber = 1

for column in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.distplot(kd[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[40]:


plt.figure(figsize = (20, 15))
plotnumber = 1

for column in cat_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.countplot(kd[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[ ]:





# In[41]:


plt.figure(figsize = (15,8))
sns.heatmap(kd.corr(), annot=True, linewidth=2, linecolor = 'lightgray')
plt.show()


# In[42]:


corr_list=[]
for cols in kd:
    corr_list.append(kd['class'].corr(kd[cols]))
corr_list


# In[43]:


corr = pd.DataFrame()
corr['features']= kd.columns
corr['correlation']= corr_list
corr = corr.sort_values('correlation', ascending= False)


# In[44]:


corr['abs_correlation'] = np.abs(corr.correlation)
corr = corr.sort_values('abs_correlation',  ascending= False)
corr


# In[45]:


plt.plot(corr.features,corr.abs_correlation, color= 'green' )
plt.title('correlation plot of classification of ckd and features')
plt.xlabel('features')
plt.ylabel('correlation')
plt.xticks(rotation=85);


# In[46]:


def voilin(col):
    fig  = px.violin(kd, y=col, x='class', color='class', box=True)
    return fig.show()



# In[47]:


voilin('specific_gravity')


# In[48]:


def kde(col):
    grid = sns.FacetGrid(kd, hue='class', height = 6, aspect = 2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    return grid


# In[49]:


kde('red_blood_cell_count')


# In[50]:


def scatter_plot(col1, col2):
    fig  = px.scatter(kd, x=col1, y=col2, color="class")
    return fig.show()


# In[51]:


scatter_plot('age', 'blood_pressure')


# In[52]:


kd.shape


# In[53]:


kd.columns


# In[54]:


kd.head()


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


train_kd, test_kd = train_test_split(kd, test_size= .25)
train_kd_x = train_kd.iloc[: , 0:-1]
train_kd_y = train_kd.iloc[: , -1]

test_kd_x = test_kd.iloc[: , 0:-1]
test_kd_y = test_kd.iloc[: , -1]


# In[57]:


from sklearn.metrics import confusion_matrix , classification_report,roc_auc_score, roc_curve , accuracy_score


# ### Model 1 
# ## Logistic Regression

# In[58]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[59]:


logreg.fit(train_kd_x,train_kd_y)


# In[60]:


pred_test = logreg.predict(test_kd_x)


# In[61]:


tab1 = confusion_matrix(test_kd_y,pred_test )
tab1


# In[62]:


print(classification_report(test_kd_y,pred_test))


# In[63]:


logred_acc = tab1.diagonal().sum()/tab1.sum()
logred_acc


# In[64]:


pred_test_prob = logreg.predict_proba(test_kd_x)


# In[65]:


roc_auc_score(test_kd_y, pred_test_prob[: , 1])


# In[66]:


fpr , tpr , thre = roc_curve(test_kd_y, pred_test_prob[: , 1])
plt.plot(fpr , tpr , color = 'green')
plt.title(" ROC on kidney disease prediction")
plt.xlabel("Fpr")
plt.ylabel("Tpr")
plt.grid()


# In[67]:


logred_acc = tab1.diagonal().sum()/tab1.sum()
logred_acc


# In[ ]:





# In[ ]:





# ### Model 2
# ## Decision Tree

# In[68]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion= 'gini',
 max_depth= 10,
 max_features='sqrt',
 min_samples_leaf= 3,
 min_samples_split= 5,
 splitter= 'random')


# In[69]:


dt.fit(train_kd_x,train_kd_y)


# In[70]:


pred_test = dt.predict(test_kd_x)


# In[71]:


tab2 = confusion_matrix(test_kd_y,pred_test)
tab2


# In[72]:


print(classification_report(test_kd_y,pred_test))


# In[73]:


dt_acc = tab2.diagonal().sum()/tab2.sum()
dt_acc


# In[74]:


dt.feature_importances_


# In[75]:


train_kd_x.columns


# In[76]:


df = pd.DataFrame()
df['Features'] = train_kd_x.columns
df['imp'] = dt.feature_importances_ 
df =df.sort_values(['imp'], ascending= False)
df # highest the value more significant is the variable


# In[77]:


df.imp[0:4].sum()


# In[78]:


df.Features[0:4]


# In[79]:


plt.bar(df.Features, df.imp , color= 'green')
plt.title('imporatant features')
plt.xlabel('features')
plt.ylabel('importance')
plt.xticks(rotation = 75);


# In[80]:


from sklearn.model_selection import GridSearchCV

GRID_PARAMETER = {
    'criterion':['gini','entropy'],
    'max_depth':[3,5,7,10],
    'splitter':['best','random'],
    'min_samples_leaf':[1,2,3,5,7],
    'min_samples_split':[1,2,3,5,7],
    'max_features':['auto', 'sqrt', 'log2']
}

grid_search_dt= GridSearchCV(dt, GRID_PARAMETER, cv=5, n_jobs=-1, verbose = 1)
grid_search_dt.fit(train_kd_x, train_kd_y)


# In[81]:


grid_search_dt.best_params_


# In[82]:


grid_search_dt.best_score_


# In[83]:


dt_acc = tab2.diagonal().sum()/tab2.sum()
dt_acc


# ### model 3
# ## SVM

# In[84]:


from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
svc = SVC( probability=True)
parameter = {
    'gamma':[0.0001, 0.001, 0.01, 0.1],
    'C':[0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameter)


# In[85]:


grid_search.fit(train_kd_x, train_kd_y)


# In[86]:


svc = SVC(kernel='linear', gamma = 0.0001, C  = 15, probability=True)

svc.fit(train_kd_x,train_kd_y)


# In[87]:


pred_test = svc.predict(test_kd_x)


# In[88]:


tab3 = confusion_matrix(test_kd_y,pred_test)
tab3


# In[89]:


print(classification_report(test_kd_y,pred_test))


# In[90]:


svm_acc = tab3.diagonal().sum()/tab3.sum()
svm_acc


# In[ ]:





# ### model 4
# ## Random Forest

# In[91]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion = "gini", max_depth = 10, max_features="sqrt", min_samples_leaf= 1, min_samples_split= 7, n_estimators = 400)


# In[92]:


rfc.fit(train_kd_x,train_kd_y)


# In[93]:


pred_test = rfc.predict(test_kd_x)


# In[94]:


tab4 = confusion_matrix(test_kd_y,pred_test)
tab4


# In[95]:


print(classification_report(test_kd_y,pred_test))


# In[96]:


rfc_acc = tab4.diagonal().sum()/tab4.sum()
rfc_acc


# In[97]:


rfc.feature_importances_


# In[98]:


df = pd.DataFrame()
df['Features'] = train_kd_x.columns
df['imp'] = rfc.feature_importances_ 
df =df.sort_values(['imp'], ascending= False)
df # highest the value more significant is the variable


# In[99]:


plt.bar(df.Features, df.imp , color= 'green')
plt.title('imporatant features')
plt.xlabel('features')
plt.ylabel('importance')
plt.xticks(rotation = 75);


# In[ ]:





# ### Cross Validation 

# In[100]:


from sklearn.model_selection import cross_val_score


# In[101]:


dt_cross_val = cross_val_score(dt,train_kd_x,train_kd_y,cv= 10,scoring='accuracy')


# In[102]:


list(dt_cross_val)


# In[103]:


dt_cross_val.mean()


# In[104]:


dt_cross_val.min()


# In[105]:


dt_cross_val.max()


# In[106]:


rfc_cross_val = cross_val_score(rfc, train_kd_x,train_kd_y,cv= 5,scoring='accuracy')


# In[107]:


list(rfc_cross_val)


# In[108]:


rfc_cross_val.mean()


# In[109]:


rfc_cross_val.min()


# In[110]:


rfc_cross_val.max()


# In[ ]:





# ### Model 5
# ## KNN

# In[111]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[112]:


knn.fit(train_kd_x,train_kd_y)


# In[113]:


#pred_test = knn.predict(test_kd_x)


# In[ ]:





# In[114]:


#pip install --upgrade numpy


# In[ ]:





# ### Model 6
# ## Gradient Boosting

# In[115]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(learning_rate= 0.1, loss = 'log_loss', n_estimators = 100)


# In[116]:


gbc.fit(train_kd_x,train_kd_y)


# In[117]:


pred_test = gbc.predict(test_kd_x)


# In[118]:


tab6=confusion_matrix(test_kd_y , pred_test)
tab6


# In[119]:


print(classification_report(test_kd_y , pred_test))


# In[120]:


gb_acc = tab6.diagonal().sum()/tab6.sum()
gb_acc


# In[121]:


gbc = GradientBoostingClassifier()

PARAMETERS = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate':[0.001, 0.1, 1, 10],
    'n_estimators':[100,150,180, 200]
}
grid_search_gbc = GridSearchCV(gbc, PARAMETERS, cv=5, n_jobs=-1, verbose= 1)
grid_search_gbc.fit(train_kd_x, train_kd_y)


# In[122]:


print(grid_search_gbc.best_params_)


# In[123]:


print(grid_search_gbc.best_score_)


# In[124]:


gb_acc = tab6.diagonal().sum()/tab6.sum()
gb_acc


# ### Model 7
# ## AdaBoost

# In[125]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()


# In[126]:


abc.fit(train_kd_x, train_kd_y)


# In[127]:


tab7 = confusion_matrix(test_kd_y , pred_test)
tab7


# In[128]:


print(classification_report(test_kd_y , pred_test))


# In[129]:


ada_acc = tab7.diagonal().sum()/tab7.sum()
ada_acc


# In[ ]:





# ### adaboost using DT

# In[130]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(dt)


# In[131]:


abc.fit(train_kd_x, train_kd_y)


# In[132]:


pred_test = abc.predict(test_kd_x)


# In[133]:


tab8= confusion_matrix(test_kd_y , pred_test)
tab8


# In[134]:


print(classification_report(test_kd_y , pred_test))


# In[135]:


adab_acc = tab8.diagonal().sum()/tab8.sum()
adab_acc


# ### adaboost using RF

# In[136]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(rfc)


# In[137]:


abc.fit(train_kd_x, train_kd_y)


# In[138]:


pred_test = abc.predict(test_kd_x)


# In[139]:


confusion_matrix(test_kd_y , pred_test)


# In[140]:


print(classification_report(test_kd_y , pred_test))


# In[ ]:





# ## model comparison

# In[141]:


models = pd.DataFrame({
    'Model':['Logistic Regression', 'DT', 'SVM', 'Random Forest Classifier','Gradient Boosting', 'adaboost'],
    'Score':[logred_acc, dt_acc, svm_acc, rfc_acc, gb_acc, ada_acc]
})

models.sort_values(by='Score', ascending = False)


# In[ ]:





# In[142]:


from sklearn import metrics
plt.figure(figsize=(8,5))
models = [
{
    'label': 'LR',
    'model': logreg,
},
{
    'label': 'DT',
    'model': dt,
},
{
    'label': 'SVM',
    'model': svc,
},

{
    'label': 'RF',
    'model': rfc,
},
{
    'label': 'GBDT',
    'model': gbc,
},
{
    'label': 'adaB',
    'model': abc,
}
    
]
for m in models:
    model = m['model'] 
    model.fit(train_kd_x, train_kd_y) 
    y_pred=model.predict(test_kd_x) 
    fpr1, tpr1, thresholds = metrics.roc_curve(test_kd_y, model.predict_proba(test_kd_x)[:,1])
    auc = metrics.roc_auc_score(test_kd_y,model.predict(test_kd_x))
    plt.plot(fpr1, tpr1, label='%s - ROC (area = %0.2f)' % (m['label'], auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.title('ROC - Kidney Disease Prediction', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.savefig("roc_kidney.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()


# In[143]:


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
models = [
{
    'label': 'LR',
    'model': logreg,
},
{
    'label': 'DT',
    'model': dt,
},
{
    'label': 'SVM',
    'model': svc,
},

{
    'label': 'RF',
    'model': rfc,
},
{
    'label': 'GBDT',
    'model': gbc,
},
{
    'label': 'adaB',
    'model': abc,
}
    
]
means_roc = []
means_accuracy = [100*round(logred_acc,4), 100*round(dt_acc,4), 100*round(svm_acc,4), 100*round(rfc_acc,4), 
                  100*round(gb_acc,4), 100*round(ada_acc,4)]

for m in models:
    model = m['model'] 
    model.fit(train_kd_x, train_kd_y) 
    y_pred=model.predict(test_kd_x) 
    fpr1, tpr1, thresholds = metrics.roc_curve(test_kd_y, model.predict_proba(test_kd_x)[:,1])
    auc = metrics.roc_auc_score(test_kd_y,model.predict(test_kd_x))
    auc = 100*round(auc,4)
    means_roc.append(auc)
 
    print(means_accuracy)
    print(means_roc)
    
n_groups = 6
means_accuracy = tuple(means_accuracy)
means_roc = tuple(means_roc)


fig, ax = plt.subplots(figsize=(5,5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_accuracy, bar_width,
alpha=opacity,
color='mediumpurple',
label='Accuracy (%)')

rects2 = plt.bar(index + bar_width, means_roc, bar_width,
alpha=opacity,
color='rebeccapurple',
label='ROC (%)')

plt.xlim([-1, 8])
plt.ylim([45, 104])

plt.title('Performance Evaluation - Kidney Disease Prediction', fontsize=12)
plt.xticks(index, ('LR', 'DT', 'SVM', 'RF' , 'GBDT', 'adaB'), rotation=40, ha='center', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.savefig("PE_kidney.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show();


# In[ ]:





# In[ ]:





# In[ ]:




