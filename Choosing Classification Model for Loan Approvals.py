
pip install plotly




import pandas as pd
import numpy as np


# # DataSet




df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv")
df.head()


# # Transforming Data

# ### Converting the effective_date & due_date columns to date format.



df['effective_date'] = pd.to_datetime(df['effective_date'])




df['due_date'] = pd.to_datetime(df['due_date'])


# **Getting respected week days from effective dates.**

# In[6]:


df['weekday'] = df['effective_date'].dt.dayofweek



df.head()


# ### Setting a threshold for weekday.  
# *1 for (days>3) and 0 for (days<4)*

# In[8]:


df['days_of_week'] = df['weekday'].apply(lambda x: 1 if (x>3) else 0)


# In[9]:


df.head()


# ###  Converting Gender..... not real gender of human, just variable   
# **male = 1 & female =0**

# In[10]:


df['Gender'] = df['Gender'].apply(lambda x:1 if(x=='male') else '0')


# In[11]:


df.head()


# In[12]:


df['loan_status'] = df['loan_status'].apply( lambda x : 1 if(x=='PAIDOFF') else 0)


# In[13]:


y = df.groupby(['education'])['loan_status'].value_counts(normalize= True)


# In[14]:


import plotly.express as px


# In[15]:


max, min  = df['weekday'].max(), df['weekday'].min()
fig = px.bar(df, x='education',color='loan_status', width=500, height =400)
fig.show()


# **Here Master or Above is insignificant**   
# ### One Hot Encoding >Eduction

# In[16]:


encoded = pd.get_dummies(df['education'])
encoded = encoded.drop('Master or Above', axis = 1)
encoded.head()


# ### Concat Two dataframe

# In[17]:


df = pd.concat([df,encoded], axis= 1)


# In[18]:


df.head()


# In[88]:


fig = px.histogram(df, y= 'Principal', color= 'loan_status',width=500, height =400)
fig.show()


# In[20]:


Features = list(df.columns)
Features = Features[3:]
Features = df[Features]


# In[89]:


label = df['loan_status']


# In[22]:


Features.drop(['due_date','effective_date','education','weekday'] ,axis= 1, inplace= True)
Features.head()


# # Normalizing the features Bcz we r going to use Euclidian Distance Matrix

# In[23]:


from sklearn import preprocessing
Features = preprocessing.StandardScaler().fit(Features).transform(Features)


# In[24]:


Features.shape


# In[25]:


from sklearn.model_selection import train_test_split as split

train_x, test_x, train_y, test_y = split(Features,label, test_size = 0.1, random_state = 1)


# # *So if you looked at all above codes then you have known almost everything in data prepration for this project & I just wants to say to you....*  
# 

# <img src ='https://pbs.twimg.com/media/Eh9Z5VMWsAUclr3?format=jpg&name=medium' width='800'>

# ** **

# ** **

# ** **

# ** **

# # Machine Learning Models

# ## K-Nearest Neighbours

# <img src ='https://charterforcompassion.org/images/menus/communities/goodneighbor.jpg'>

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[27]:


ks =20
mean_acc = np.zeros((ks-1))
std_acc = np.zeros((ks-1))
for n in range(1, ks):
    neigh = KNeighborsClassifier(n_neighbors= n)
    neigh.fit(train_x, train_y)
    prediction = neigh.predict(test_x)
    mean_acc[n-1] = metrics.accuracy_score(test_y, prediction)
    
mean_acc = list(mean_acc)

for i in range(len(mean_acc)):
    mean_acc[i] = round(mean_acc[i]*100,1) 
    


# ### Plotting Accuracy

# In[28]:


import plotly.express as px
fig = px.line(x = np.arange(1,20,1), y= mean_acc,title ='Acurracy v/s K-value',text = mean_acc)
fig.show()

# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')


# **Here we note that accuracy incease upto 83.3% with k= 3.**  
# *Therefor we will build model with K = 3*

# In[29]:


del mean_acc


# In[30]:


KNN = KNeighborsClassifier(n_neighbors= 5)
KNN.fit(train_x, train_y)


# ## Testing Model

# In[31]:


predict = KNN.predict(test_x)


# ## Jaccard Score

# In[32]:


from sklearn.metrics import jaccard_score
score = jaccard_score(test_y,predict)
print('Jaccard-Score for KNN with [K=10] is {:.2f}%'.format(score*100))


# ## F1 Score 

# In[33]:


from sklearn.metrics import f1_score
score = f1_score(test_y, predict)
print('F1-Score for KNN with [K=10] is {:.2f}%'.format(score*100))


# ** **

# ## Decision Tree

# <img src='https://files.ai-pool.com/a/aba6fb4dcd4c3d01372085631f47d122.png'>

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


n = 6
mean_acc = np.zeros(n)
for i  in range(1,n+1):
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(train_x, train_y)
    prediction = tree.predict(test_x)
    mean_acc[i-1] = metrics.accuracy_score(test_y, prediction)

mean_acc = list(mean_acc)
for i in range(len(mean_acc)):
    mean_acc[i]  = round(mean_acc[i], 2)


# In[36]:


fig = px.line(x=np.arange(1,n+1,1), y= list(mean_acc), text=list(mean_acc), title='Accuracy v/s Layer Number')
fig.show()


# **Here we see max acurracy is with *[Layer = 2]* after that acurracy is in harmonic trend**

# In[37]:


tree = DecisionTreeClassifier(criterion ='entropy')
tree.fit(train_x, train_y)
prediction = tree.predict(test_x)


# ## Testing Model

# ### Jaccard_score

# In[38]:


score = jaccard_score(test_y,prediction)
print('Jaccard-Score for Tree is {:.2f}%'.format(score*100))


# ### F1-Score

# In[39]:


score = f1_score(test_y, prediction)
print('F1-Score for Tree is {:.2f}%'.format(score*100))


# ** ** 

# # Support Vector Machine

# <img src='https://i2.wp.com/dataaspirant.com/wp-content/uploads/2017/01/Support-vector-machine-svm.jpg?resize=768%2C576&ssl=1'>

# In[40]:


from sklearn import svm
machine = svm.SVC()
machine.fit(train_x, train_y)
prediction = machine.predict(test_x)


# ### Jaccard-Score

# In[41]:


score = jaccard_score(test_y,prediction)
print('Jaccard-Score for SVM is {:.2f}%'.format(score*100))


# ### F1-Score

# In[42]:


score = f1_score(test_y, prediction)
print('F1-Score for SVM is {:.2f}%'.format(score*100))


# ** ** 

# # Logistic Regression

# <img src="https://www.tibco.com/sites/tibco/files/media_entity/2020-09/logistic-regression-diagram.svg">

# In[43]:


from sklearn.linear_model import LogisticRegression
linearModel = LogisticRegression(C=0.01, solver = 'liblinear')
linearModel.fit(train_x, train_y)
prediction = linearModel.predict(test_x)


# ### Jaccard-Score

# In[44]:


score = jaccard_score(test_y,prediction)
print('Jaccard-Score for Logistics is {:.2f}%'.format(score*100))


# ### F1-Score

# In[45]:


score = f1_score(test_y, prediction)
print('F1-Score for Logistics is {:.2f}%'.format(score*100))


# ### LogLoss

# In[46]:


from sklearn.metrics import log_loss

score = log_loss(test_y,prediction)
print('LogLoss-Score for Logistics is:',score)


# # Testing Models

# In[74]:


test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')
test_df.head()


# In[75]:


# Removing unwanted first two columns.
remove = test_df.columns[0:2]
test_df = test_df.drop(remove, axis=1)

#Converting Loan status from descriptive classification to binary classification.
test_df['loan_status'] = test_df['loan_status'].apply(lambda x: 1 if(x=='PAIDOFF') else 0)

#Converting Date Times in  date columns, so that we could do some feature engineerign on it later.
test_df['effective_date'] = pd.to_datetime(df['effective_date'])

#Getting Days of week on every date.
test_df['weekday'] = test_df['effective_date'].dt.dayofweek
#Setting threshold for days of week by 1 for x>3 and 0 for x<4.
test_df['days_of_week'] = test_df['weekday'].apply(lambda x: 1 if(x>3) else 0)


#Conversion of discriptive classification to binary classififcation.
test_df['Gender'] = test_df['Gender'].apply(lambda x: 1 if (x=='Male') else 0)

# One Hot Encoding for eduction columns.
dummies = pd.get_dummies(test_df['education'])

# Concatenating two dataframe of dummy and test_df.
test_df = pd.concat([test_df, dummies], axis = 1)

#Dropping unnecessasry columns.
test_df.drop(test_df.iloc[:,[3,4,6,8,12]],axis  = 1, inplace = True)

#Seprating test_y columns for checking accuracy of model.
labels = test_df['loan_status']
test_df.drop(['loan_status'], axis = 1, inplace = True)

# Normalizing our DataFrame.
Normalize = preprocessing.StandardScaler()
Normalize = Normalize.fit(test_df)
test_X = Normalize.transform(test_df)

# Our Data Cake is ready to serve to our Hungary Machine Learning Models.


# # Start Testing ML-Models

# <img src ='https://www.testim.io/wp-content/uploads/2019/11/Testim-What-is-a-Test-Environment_-A-Guide-to-Managing-Your-Testing-A.png' width= "400">

# ### Jaccard Score

# In[84]:


t1 = "Jaccard-Score"
t2 = 'F1-Score'
t3 = "LogLoss"
Names = ['KNN','Tree','SVM','Logistics']
#Listing all models 
models = [KNN,tree, machine, linearModel]

# Creating a dictionary for saving Scores as Dataframe.
result = {'Name':[], t1:[], t2:[], t3:[]}

# scores.append (jaccard_score(test_y,predict))
# scores.append (f1_score(test_y, prediction))
# scores.append( log_loss(test_y,prediction))
for i in range(4):
    prediction = models[i].predict(test_X)
    Score1 = jaccard_score(labels, prediction)
    Score1 = "{:,.2f}%".format(Score1*100)
    
    Score2 = f1_score(labels, prediction)
    Score2 = "{:,.2f}%".format(Score2*100)
    
    Score3 = log_loss(labels, prediction)
    Score3 = "{:,.2f}".format(Score3)
    
    result['Name'].append(Names[i])
    result[t1].append(Score1)
    result[t2].append(Score2)
    result[t3].append(Score3)
    
Test_Result = pd.DataFrame.from_dict(result)


# # Result

# In[87]:


Test_Result


# In[ ]:




