#!/usr/bin/env python
# coding: utf-8

# # **Loading Data and Importing Modules**
# 

# In[1]:


#importing the required packages
import pandas as pd
import numpy as np
#import visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix,roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


# In[2]:


st.title('Telecom Churn Prediction')
st.sidebar.header("'1' means YES and '0' means NO ")
st.sidebar.header('User Input Parameters')


# In[3]:


data= pd.read_csv('churn.csv', index_col = 0)


# In[4]:


data


# # **Understand More About The Data**

# In[5]:


# Viewing the data of top 5 rows to loo the glimps of the data
data.head(5)


# In[6]:


# View the data of bottom 5 rows to look the glimps of the data
data.tail(5)


# In[7]:


#Getting the shape of dataset with rows and columns
print(data.shape)


# In[8]:


#Getting all the columns
print("Features of the dataset:")
data.columns


# **Breakdown of Our Features:**
# 
# ●	state: Categorical, for the 51 states and the District of Columbia.
# 
# ●	Area.code
# 
# ●	account.length: how long the account has been active.
# 
# ●	voice.plan: yes or no, voicemail plan.
# 
# ●	voice.messages: number of voicemail messages.
# 
# ●	intl.plan: yes or no, international plan.
# 
# ●	intl.mins: minutes customer used service to make international calls.
# 
# ●	intl.calls: total number of international calls.
# 
# ●	intl.charge: total international charge.
# 
# ●	day.mins: minutes customer used service during the day.
# 
# ●	day.calls: total number of calls during the day.
# 
# ●	day.charge: total charge during the day.
# 
# ●	eve.mins: minutes customer used service during the evening.
# 
# ●	eve.calls: total number of calls during the evening.
# 
# ●	eve.charge: total charge during the evening.
# 
# ●	night.mins: minutes customer used service during the night.
# 
# ●	night.calls: total number of calls during the night.
# 
# ●	night.charge: total charge during the night.
# 
# ●	customer.calls: number of calls to customer service.
# 
# ●	churn: Categorical, yes or no. Indicator of whether the customer has left the company (yes or no).
# 

# In[9]:


#Getting the data types of all the columns
data.dtypes


# In[10]:


data['day.charge']=data['day.charge'].astype('float64')
data['eve.mins']=data['eve.mins'].astype('float64')         


# In[11]:


#check details about the data set
data.info()
#we see that we have 3333 entries and no null values are present


# In[12]:


data.nunique()


# In[13]:


#Looking for the description of the dataset to get insights of the data
data.describe(include='all')


# In[14]:


#Printing the count of true and false in 'churn' feature
print(data.churn.value_counts())


# ## Checking for Missing And Duplicate values

# In[15]:


#check for count of missing values in each column.
data.isna().sum()
data.isnull().sum()
#as we see there are no missing values present in nay column.


# In[16]:


data['day.charge'].fillna(data['day.charge'].mean(), inplace=True)


# In[17]:


data['eve.mins'].fillna(data['eve.mins'].mean(), inplace=True)


# In[18]:


missing = pd.DataFrame((data.isnull().sum())*100/data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0, data=missing)
plt.xticks(rotation =90,fontsize =11)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# In[19]:


# Checking Duplicate Values
len(data[data.duplicated()])


# ***As of now There are 5000 rows and 20 columns in above dataset.***
# 
# ***out of which there are 5 object data type i.e voice.plan, intl.plan, churn***
# 
# ***8 float data type***, 
# 
# ***7integer data type,*** 
# 
# ***There are 2 missing values so need to do the missing value imputation,***
# 
# ***And there are no duplicate value present.*** 

# In[20]:


df_input = user_input_features()
st.subheader("User input parameters")
st.write(df_input)


# # **Exploratory Data Analysis Of The Data Set**

# ### Analyzing What The Dependent Variable Said To Us i.e 'CHURN'.

# In[ ]:


#convert string values(yes and no) of churn column to 1 and 0
data.loc[data.churn=='no', 'churn']=0
data.loc[data.churn=='yes', 'churn']=1
#convert to integer
data['churn']=data['churn'].astype('int32')


# In[ ]:


#Printing the unique value inside "churn" column
data["churn"].unique()


# In[ ]:


#Printing the count of true and false in 'churn' feature
print(data.churn.value_counts())


# In[ ]:


100*data['churn'].value_counts()/len(data['churn'])


# In[ ]:


#To get the pie chart to analyze churn
data['churn'].value_counts().plot.pie(explode=[0.05,0.05], autopct='%1.1f%%',  startangle=90,shadow=True, figsize=(8,8))
plt.title('Pie Chart for Churn')
plt.show()


# In[ ]:


#let's see churn by using countplot
sns.countplot(x=data.churn)


# In[ ]:


sns.boxplot(x=data['churn'])


# ***After analyzing the churn column, we had little to say like almost 15% of customers have churned. let's see what other features say to us and what relation we get after correlated with churn***

# ### Analyzing State Column

# In[ ]:


#printing the unique value of sate column
data['state'].nunique()


# In[ ]:


#Comparison churn with state by using countplot (barchart)
sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
ax = sns.countplot(x='state', hue="churn", data=data)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (12, 7)
color = plt.cm.copper(np.linspace(0, 0.5, 20))
((data.groupby(['state'])['churn'].mean())*100).sort_values(ascending = False).head(6).plot.bar(color = ['violet','indigo','b','g','y','orange','r'])
plt.title(" State with most churn percentage", fontsize = 20)
plt.xlabel('state', fontsize = 15)
plt.ylabel('percentage', fontsize = 15)
plt.show()


# In[ ]:


#calculate State vs Churn percentage
State_data = pd.crosstab(data["state"],data["churn"])
State_data['Percentage_Churn'] = State_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(State_data)


# In[ ]:


#show the most churn state of top 10 by ascending the above list
data.groupby(['state'])['churn'].mean().sort_values(ascending = False).head(10)


# ***There is 51 unique state present who have different churn rate.*** 
# 
# ***From the above analysis CA, NJ, WA, TX, MT, MD, NV, ME, KS, OK are the ones who have a higher churn rate of more than 21.***
# 
#  ***The reason for this churn rate from a particular state may be due to the low coverage of the cellular network.***

# ### Analyzing "Area Code" column

# In[ ]:


#calculate Area code vs Churn percentage
Area_code_data = pd.crosstab(data["area.code"],data["churn"])
Area_code_data['Percentage_Churn'] = Area_code_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Area_code_data)


# In[ ]:


sns.set(style="darkgrid")
ax = sns.countplot(x='area.code', hue="churn", data=data)
plt.show()


# ***In the above data, we notice that there is only 3 unique value are there i.e 408,415,510 and the churn rate of these area codes are almost same.***
# 
#  ***we don't think there is any kind of relation present between the "area code" and"churn" due to which the customer leaves the operator.***

# ### STATE and AREA.CODE
# 
#  > By analysing the above plots and data its very difficult to predict churn customer from area code or state.
#   
#  > Both features are independent in nature. 
#   
#  > There are no use of this feature for predicting churn customer.
#   
#  > Having no relationship between churn and this features.

# # Dropping Unused features

# In[ ]:


data1= data.drop(columns=['state','area.code'],axis=1)


# In[ ]:


data1.head()


# ### Analyzing "Account length" column

# In[ ]:


#Separating churn and non churn customers
churn_df     = data1[data1["churn"] == bool(True)]
not_churn_df = data1[data1["churn"] == bool(False)]


# In[ ]:


#Account length vs Churn
sns.distplot(data1['account.length'])


# In[ ]:


#comparison of churned account length and not churned account length 
sns.distplot(data1['account.length'],color = 'purple',label="All")
sns.distplot(churn_df['account.length'],color = "red",hist=False,label="Churned")
sns.distplot(not_churn_df['account.length'],color = 'green',hist= False,label="Not churned")
plt.legend()


# In[ ]:


fig = plt.figure(figsize =(10, 8)) 
data1.boxplot(column='account.length', by='churn')
fig.suptitle('account.length', fontsize=14, fontweight='bold')
plt.show()


# ***After analyzing various aspects of the "account length" column we didn't found any useful relation to churn. so we aren't able to build any connection to the churn as of now. let's see what other features say about the churn.***

# ### Analyzing "International Plan" column

# In[ ]:


#Show count value of 'yes','no'
data1['intl.plan'].value_counts()


# In[ ]:


#Show the unique data of "International plan"
data1["intl.plan"].unique()


# In[ ]:


#Calculate the International Plan vs Churn percentage 
International_plan_data = pd.crosstab(data1["intl.plan"],data1["churn"])
International_plan_data['Percentage Churn'] = International_plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(International_plan_data)


# In[ ]:


#Analysing by using countplot
sns.countplot(x='intl.plan',hue="churn",data = data1)


# ***From the above data we get***
# 
# ***There are 4527 customers who  dont have a international plan.***
# 
# ***There are 473 customers who have a international plan.***
# 
# ***Among those who have a international plan 42.07 % people churn.***
# 
# ***Whereas among those who dont have a international plan only 11.22 % people churn.***
# 
# ***So basically the people who bought International plans are churning in big numbers.***
# 
# ***Probably because of connectivity issues or high call charge.***

# ### Analyzing "Voice Plan" column

# In[ ]:


#show the unique value of the "Voice mail plan" column
data1["voice.plan"].unique()


# In[ ]:


#Calculate the Voice Mail Plan vs Churn percentage
Voice_mail_plan_data = pd.crosstab(data1["voice.plan"], data1["churn"])
Voice_mail_plan_data['Percentage Churn'] = Voice_mail_plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Voice_mail_plan_data)


# In[ ]:


#Analysing by using countplot
sns.countplot(x='voice.plan',hue="churn",data = data1)


# ***As we can see there is are no clear relation between voice mail plan and churn so we can't clearly say anything so let's move to the next voice mail feature i.e number of voice mail, let's see what it gives to us.***

# ### Analyzing "Voice messages" column

# In[ ]:


#show the data of 'voice messages' 
data1['voice.messages'].unique()


# In[ ]:


#Printing the data of 'voice messages'
data1['voice.messages'].value_counts()


# In[ ]:


#Show the details of 'voice messages' data
data1['voice.messages'].describe()


# In[ ]:


#Analysing by using displot diagram
sns.distplot(data1['voice.messages'])


# In[ ]:


#Analysing by using boxplot diagram between 'voice messages' and 'churn'
fig = plt.figure(figsize =(10, 8)) 
data1.boxplot(column='voice.messages', by='churn')
fig.suptitle('voice message', fontsize=14, fontweight='bold')
plt.show()


# ***After analyzing the above voice message feature data we get an insight that when there are more than 20 voice-mail messages then  there is a churn***
# 
# ***For that, we need to improve the voice message quality.***

# ### Analyzing "Customer calls" column

# In[ ]:


#Printing the data of customer service calls 
data1['customer.calls'].value_counts()


# In[ ]:


#Calculating the Customer calls vs Churn percentage
Customer_service_calls_data = pd.crosstab(data1['customer.calls'],data1["churn"])
Customer_service_calls_data['Percentage_Churn'] = Customer_service_calls_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Customer_service_calls_data)


# In[ ]:


#Analysing using countplot
sns.countplot(x='customer.calls',hue="churn",data = data1)


# In[ ]:


fig = plt.figure(figsize =(10, 8)) 
data1.boxplot(column='customer.calls', by='churn')
fig.suptitle('customer.calls', fontsize=14, fontweight='bold')
plt.show()


# ***It is observed from the above analysis that, mostly because of bad customer service, people tend to leave the operator.***
# 
# ***The above data indicating that those customers who called the service center 5 times or above those customer churn percentage is higher than 60%,***
# 
# ***And customers who have called once also have a high churn rate indicating their issue was not solved in the first attempt.***
# 
# ***So operator should work to improve the service call.***

# ### Analyzing all calls minutes,all calls, all calls charge together
# ***As these data sets are numerical data type, so for analysing with the 'churn' which is a catagorical data set, We are using mean, median, and box plots.***

# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['day.calls'].mean())


# In[ ]:


sns.boxplot(x='churn', y='day.calls', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['day.mins'].mean())


# In[ ]:


sns.boxplot(x='churn', y='day.mins', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['day.charge'].mean())


# In[ ]:


sns.boxplot(x='churn', y='day.charge', data=data1)


# In[ ]:


#show the relation using scatter plot
sns.scatterplot(x="day.mins", y="day.charge", hue="churn", data=data1, palette='hls')


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['eve.calls'].mean())


# In[ ]:


sns.boxplot(x='churn', y='eve.calls', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['eve.mins'].mean())


# In[ ]:


sns.boxplot(x='churn', y='eve.mins', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['eve.charge'].mean())


# In[ ]:


sns.boxplot(x='churn', y='eve.charge', data=data1)


# In[ ]:


#show the relation using scatter plot
sns.scatterplot(x="eve.mins", y="eve.charge", hue="churn", data=data1, palette='hls')


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['night.calls'].mean())


# In[ ]:


sns.boxplot(x='churn', y='night.calls', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['night.charge'].mean())


# In[ ]:


sns.boxplot(x='churn', y='night.charge', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['night.mins'].mean())


# In[ ]:


sns.boxplot(x='churn', y='night.mins', data=data1)


# In[ ]:


#show the relation using scatter plot
sns.scatterplot(x="night.mins", y="night.charge", hue="churn", data=data1, palette='hls')


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['intl.mins'].mean())


# In[ ]:


sns.boxplot(x='churn', y='intl.mins', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['intl.calls'].mean())


# In[ ]:


sns.boxplot(x='churn', y='intl.calls', data=data1)


# In[ ]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['intl.charge'].mean())


# In[ ]:


sns.boxplot(x='churn', y='intl.charge', data=data1)


# In[ ]:


#show the relation using scatter plot
sns.scatterplot(x="intl.mins", y="intl.charge", hue="churn", data=data1, palette='hls')


# In[ ]:


#Deriving a relation between overall call charge and overall call minutes   
day_charge_perm = data1['day.charge'].mean()/data1['day.mins'].mean()
eve_charge_perm = data1['eve.charge'].mean()/data1['eve.mins'].mean()
night_charge_perm = data1['night.charge'].mean()/data1['night.mins'].mean()
int_charge_perm= data1['intl.charge'].mean()/data1['intl.mins'].mean()


# In[ ]:


print([day_charge_perm,eve_charge_perm,night_charge_perm,int_charge_perm])


# In[ ]:


sns.barplot(x=['day','Eve','night','intl'],y=[day_charge_perm,eve_charge_perm,night_charge_perm,int_charge_perm])


# 
# ***After analyzing the above dataset we have noticed that total day/night/eve minutes/call/charges are not put any kind of cause for churn rate. But international call charges are high as compare to others it's an obvious thing but that may be a cause for international plan customers to churn out.***

# ## Graphical Analysis

# ### UNIVARIATE ANALYSIS
# 
# In Univariate Analysis we analyze data over a single column from the numerical dataset, for this we use 3 types of plot which are **box plot, strip plot, dis plot.** 
# 
# 

# In[ ]:


df1=data1.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.displot(data=df1, x=column, hue='churn')
plt.show()


# In[ ]:


#Printing boxplot for each numerical column present in the data set
df1=data1.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=df1, x=column, hue='churn')
plt.show()


# In[ ]:





# In[ ]:


#Printing strip plot for each numerical column present in the data set
df1=data.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.stripplot(data=df1, x=column, hue='churn')
plt.show()


# ### converting categorical variables into dummy variables

# In[ ]:


data1_dummies = pd.get_dummies(data1)
data1_dummies.head()


# ### Multivariate Analysis
# In Multivariate Analysis we analyze data by taking more than two columns into consideration from a dataset,for this we using **correlation plot,correlation matrix, correletaion heatmap, pair plot**

# In[ ]:


#build correlation of all predictors with churn
plt.figure(figsize=(20,8))
data1_dummies.corr()['churn'].sort_values(ascending=False).plot(kind='bar')


# In[ ]:


plt.figure(figsize=(17,8))
sns.heatmap(data1_dummies.corr(), cmap="Paired")


# In[ ]:


## plot the Correlation matrix
plt.figure(figsize=(17,8))
correlation=data1_dummies.corr()
sns.heatmap(abs(correlation), annot=True, cmap='coolwarm')


# # Finding Outliers in Data

# In[ ]:


def find_outliers_IQR(data):
   q1=data.quantile(0.25)
   q3=data.quantile(0.75)
   IQR=q3-q1
   outliers = data[((data<(q1-1.5*IQR)) | (data>(q3+1.5*IQR)))]
   return outliers


# In[ ]:


outliers=find_outliers_IQR(data1)


# In[ ]:


outliers.head(20)


# In[ ]:


outliers.info()


# ## Dropping categorical data from outliers dataframe

# In[ ]:


outliers=outliers.drop(columns=['voice.plan','intl.plan','customer.calls','churn'],axis=1)


# In[ ]:


outliers.head()


# In[ ]:


outliers.info()


# In[ ]:


outlier_data=outliers[outliers.values>=0]
outlier_data


# In[ ]:


outlier_data.info()


# In[ ]:


outlier_data['churn']=data1['churn']


# In[ ]:


outlier_data


# In[ ]:


outlier_data1=outlier_data[~outlier_data.index.duplicated(keep='first')]
outlier_data1


# In[ ]:


outlier_data1.info()


# In[ ]:


print(outlier_data1.churn.value_counts())
outlier_data1['churn'].value_counts().plot(kind="pie",autopct="%1.2f%%",figsize=(5,5))


# ### There are 391 customers who are outliers but they are not churns customers
# ### And 87 customers are outliers but they are churns

# In[ ]:


data1.boxplot(figsize=(20,10))


# # Removing not churn customer outliers

# In[ ]:


nc=outlier_data1.index[outlier_data1['churn']==0]


# In[ ]:


ncco=data1.drop(nc)


# In[ ]:


ncco.info()


# # **CONCLUSION:**

# ***After performing exploratory data analysis on the data set, this is what we have incurred from data:***
# * ***There are some states where the churn rate is high as compared to others may be due to low network coverage.***
# ****Area code and Account length do not play any kind of role regarding the churn rate so,it's redundant data columns***
# ****In the International plan those customers who have this plan are churn more and also the international calling charges are also high so the customer who has the plan unsatisfied with network issues and high call charge***
# ****IN the voice mail section when there are more than 20 voice-mail messages then there is a churn so it basically means that the quality of voice mail is not good***.
# ****Total day call minutes, total day calls, Total day charge, Total eve minutes, Total eve calls,  Total eve charge, Total night minutes,  Total night calls,  Total night charge, these columns didn't play any kind of role regarding the churn rate.***
# ****In international calls data shows that the churn rate of those customers is high, those who take the international plan so it means that in international call charges are high also there is a call drop or network issue.***
# ****In Customer service calls data shows us that whenever an unsatisfied customer called the service center the churn rate is high, which means the service center didn't resolve the customer issue.*** 

# ### **RECCOMENDATIONS:**
# 
# 
# * ***Improve network coverage churned state***
# *  ***In international plan provide some discount plan to the customer***
# *  ***Improve the voice mail quality or take feedback from the customer***
# *  ***Improve the service of call center and take frequently feedback from the customer regarding their issue and try to solve it as soon as possible***
# 
# 
# 
# 

# In[ ]:


ncco.to_csv('updated_churn.csv')


# # Model building

# In[ ]:


df2 = pd.read_csv('updated_churn.csv', index_col = 0)

encode = LabelEncoder()
df2['voice.plan'] = encode.fit_transform(df2['voice.plan'])
df2['intl.plan'] = encode.fit_transform(df2['intl.plan'])

scaler = MinMaxScaler()
df2.iloc[:,[0,2,4,6,7,9,10,12,13,15,5,8,11,14]] = scaler.fit_transform(df2.iloc[:,[0,2,4,6,7,9,10,12,13,15,5,8,11,14]])

df2.head()


# In[ ]:


X = df2.drop(labels='churn', axis=1)
y = df2['churn']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, shuffle= True, random_state=27, stratify=y)


# ## **Logistic regression 

# In[ ]:


from sklearn.linear_model import LogisticRegression, RidgeClassifier
l_model = LogisticRegression(max_iter=700, random_state=27, class_weight= {0:1, 1:3})


# In[ ]:


model_grid_logic = GridSearchCV(estimator = l_model,param_grid = [{'penalty': ['l1', 'l2', 'elasticnet'],
                                                                   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                                                   'C': [100, 10, 1.0, 0.1, 0.01] }])
model_grid_logic.fit(X,y)

print('For our Logistic Regression model, Best hyperparameters are {}'.format(model_grid_logic.best_params_))
print('For our Logistic Regression model, best accuracy score is {}'.format(model_grid_logic.best_score_))


# In[ ]:


l_final = LogisticRegression(max_iter=700, random_state=27, class_weight= {0:1, 1:3},penalty= 'l2', solver='liblinear',C=0.01)
l_final.fit(X_train, y_train)


# ### Train data

# In[ ]:


y_train_predict = l_final.predict(X_train)


# In[ ]:


acc_logic_train = accuracy_score(y_train, y_train_predict)
acc_logic_train


# In[ ]:


print(classification_report(y_train, y_train_predict))


# ### Test data

# In[ ]:


y_test_predict = l_final.predict(X_test)


# In[ ]:


acc_logic_test = accuracy_score(y_test, y_test_predict)
acc_logic_test


# In[ ]:


confusion_matrix(y_test, y_test_predict)


# In[ ]:


print(classification_report(y_test, y_test_predict))


# In[ ]:


plot_confusion_matrix(estimator= l_final,X=X_test, y_true=y_test,cmap='Blues')
plt.show()


# In[ ]:





# ## **Ridge Classifier

# In[ ]:


from sklearn.linear_model import LogisticRegression, RidgeClassifier
R_model = RidgeClassifier(max_iter=700, random_state=27, class_weight= {0:1, 1:2.5})


# In[ ]:


model_grid_ridge = GridSearchCV(estimator = R_model,param_grid = [{'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] }])
model_grid_ridge.fit(X,y)

print('For our Ridge Regression model, Best hyperparameters are {}'.format(model_grid_ridge.best_params_))
print('For our Ridge Regression model, best accuracy score is {}'.format(model_grid_ridge.best_score_))


# In[ ]:


R_final = RidgeClassifier(max_iter=700, random_state=27, class_weight= {0:1, 1:2.5}, alpha=0.2)
R_final.fit(X_train, y_train)


# ### Train data

# In[ ]:


y_train_predict_R = R_final.predict(X_train)


# In[ ]:


acc_ridge_train = accuracy_score(y_train, y_train_predict_R)
acc_ridge_train


# In[ ]:


print(classification_report(y_train, y_train_predict_R))


# ### Test data

# In[ ]:


y_test_predict_R = R_final.predict(X_test)


# In[ ]:


acc_ridge_test = accuracy_score(y_test, y_test_predict_R)
acc_ridge_test


# In[ ]:


confusion_matrix(y_test, y_test_predict_R)


# In[ ]:


print(classification_report(y_test, y_test_predict_R))


# In[ ]:


plot_confusion_matrix(estimator= R_final,X=X_test, y_true=y_test,cmap='Blues')
plt.show()


# ##  **Random Forest classifier

# In[ ]:


dt_model = RandomForestClassifier(n_estimators=500, random_state=27)
dt_model.fit(X,y)


# In[ ]:


model_grid = GridSearchCV(estimator = dt_model,param_grid = [{'criterion':['gini','entropy'],'max_depth': [2,3,4,5,6]}])
model_grid.fit(X,y)

print('For our RandomForest model, Best hyperparameters are {}'.format(model_grid.best_params_))
print('For our RandomForest model, best accuracy score is {}'.format(model_grid.best_score_))


# In[ ]:


dt_final = RandomForestClassifier(n_estimators=500, criterion= 'entropy', max_depth=6, random_state=27,class_weight= {0:1, 1:3.5})
dt_final.fit(X_train,y_train)


# ### Train data

# In[ ]:


y_train_pred_dt = dt_final.predict(X_train)


# In[ ]:


acc_random_train = accuracy_score(y_train,y_train_pred_dt)


# In[ ]:


print(classification_report(y_train, y_train_pred_dt))


# ### Test data

# In[ ]:


y_test_pred_dt = dt_final.predict(X_test)


# In[ ]:


acc_random_test = accuracy_score(y_test,y_test_pred_dt)


# In[ ]:


print(classification_report(y_test,y_test_pred_dt))


# In[ ]:


plot_confusion_matrix(estimator= dt_final,X=X_test, y_true=y_test,cmap='Blues')
plt.show()


# ## **Decision Tree classifier

# In[ ]:


dtt_model = DecisionTreeClassifier(random_state=27, class_weight= {1:1,0:3.5})
dtt_model.fit(X_train,y_train)


# In[ ]:


grid_model = GridSearchCV(estimator = dtt_model, param_grid = [{'criterion': ["gini", "entropy"],'max_depth': [1,2,3,4,5]}])
grid_model.fit(X,y)


print('Best criterian for our decision tree model is: ', grid_model.best_params_)
print('Best Score for our decision tree model is: ', grid_model.best_score_)


# In[ ]:


dtt1_model = DecisionTreeClassifier(criterion='entropy', max_depth=5,class_weight= {1:1,0:3.5})
dtt1_model.fit(X_train,y_train)


# ### Train data

# In[ ]:


y_train_pred_dtt1 = dtt1_model.predict(X_train)


# In[ ]:


acc_decision_train = accuracy_score(y_train, y_train_pred_dtt1)


# In[ ]:


confusion_matrix(y_train, y_train_pred_dtt1)


# In[ ]:


print(classification_report(y_train, y_train_pred_dtt1))


# In[ ]:


rc_sc = roc_auc_score(y_train, y_train_pred_dtt1)


# In[ ]:


fpr, tpr, thres = roc_curve(y_train, y_train_pred_dtt1)


# In[ ]:


plt.plot(fpr,tpr,'go--', linewidth=2, markersize=12)
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive rate (1-True Positive rate)')
plt.ylabel('True Positive rate')
plt.show()


# ### Test data

# In[ ]:


y_test_pred_dtt1 = dtt1_model.predict(X_test)


# In[ ]:


acc_decision_test = accuracy_score(y_test, y_test_pred_dtt1)


# In[ ]:


confusion_matrix(y_test, y_test_pred_dtt1)


# In[ ]:


print(classification_report(y_test, y_test_pred_dtt1))


# In[ ]:


rc_sc1 = roc_auc_score(y_test, y_test_pred_dtt1)


# In[ ]:


fpr1, tpr1, thres1 = roc_curve(y_test, y_test_pred_dtt1)


# In[ ]:


plt.plot(fpr1,tpr1,'go--', linewidth=2, markersize=12)
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive rate (1-True Positive rate)')
plt.ylabel('True Positive rate')
plt.show()


# ## Accuracy score comparison

# In[ ]:


Acc = pd.DataFrame(data = {'Algorithm': ['Logistic Regression','Ridge regression','Decision Tree','Random Forest'],
                           'Train accuracy': [acc_logic_train,acc_ridge_train, acc_decision_train,acc_random_train],
                           'Test accuracy': [acc_logic_test,acc_ridge_test,  acc_decision_test,acc_random_test]})

Acc


# # Over and Under Sampling

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=1)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)


ax = y_train_rus.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Under-sampling")


# In[ ]:


y_train_rus.value_counts()


# In[ ]:





# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy="not majority")
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)


ax = y_train_ros.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling")


# In[ ]:


y_train_ros.value_counts()


# # Model building without class balancing 

# In[ ]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#training cross-validate models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = RandomForestClassifier(random_state=42)
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring)

#apply model to make predictions
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

#display model performance results
df_labels = pd.Series(['MCC_train','MCC_CV','MCC_test'], name = 'Performance_metric_names')
df_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
df1 = pd.concat([df_labels, df_values], axis=1)
df1


# # Model building with undersampling data

# In[ ]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train_rus, y_train_rus)

#training cross-validate models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = RandomForestClassifier(random_state=42)
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train_rus, y_train_rus, cv=5, scoring=cv_scoring)

#apply model to make predictions
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train_rus)
y_test_pred = model.predict(X_test)

mcc_train = matthews_corrcoef(y_train_rus, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

#display model performance results
df_labels = pd.Series(['MCC_train','MCC_CV','MCC_test'], name = 'Performance_metric_names')
df_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
df2 = pd.concat([df_labels, df_values], axis=1)
df2


# # Model with balanced data

# In[ ]:


model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

#training cross-validate models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = RandomForestClassifier(random_state=42, class_weight='balanced')
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring)

#apply model to make predictions
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

#display model performance results
df_labels = pd.Series(['MCC_train','MCC_CV','MCC_test'], name = 'Performance_metric_names')
df_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
df4 = pd.concat([df_labels, df_values], axis=1)
df4


# # Model with over sampling data

# In[ ]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train_ros, y_train_ros)

#training cross-validate models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = RandomForestClassifier(random_state=42)
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train_ros, y_train_ros, cv=5, scoring=cv_scoring)

#apply model to make predictions
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train_ros)
y_test_pred = model.predict(X_test)

mcc_train = matthews_corrcoef(y_train_ros, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

#display model performance results
df_labels = pd.Series(['MCC_train','MCC_CV','MCC_test'], name = 'Performance_metric_names')
df_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
df3 = pd.concat([df_labels, df_values], axis=1)
df3


# In[ ]:


print(classification_report(y_train_ros, y_train_pred))


# In[ ]:


print(classification_report(y_test, y_test_pred))


# In[ ]:


plot_confusion_matrix(estimator= model,X=X_test, y_true=y_test,cmap='Blues')
plt.show()


# In[ ]:


df = pd.DataFrame({'Actual':y_test, 'Predicted':y_test_pred})
df


# In[ ]:


df5 = pd.concat([df1.Performance_metric_values,
               df2.Performance_metric_values,
               df3.Performance_metric_values,
               df4.Performance_metric_values], axis=1)

df5.columns = ['No class balancing', 'Class balancing(undersampling)', 'Class balancing(oversampling)', 'Class balancing(class weights)']
df5 = df5.T
df5.columns = ['Training', 'CV', 'Test']
df5


# In[ ]:


# Prepare the model
array = df2.values
X = array[:, 0:18]
Y = array[:, -1]
validation_size = 0.25
seed = 7
X_train, X_validation, y_train, y_validation = train_test_split(X_train_ros, y_train_ros, test_size=validation_size, random_state=seed)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = RandomForestClassifier(random_state=42)
model.fit(rescaledX, y_train)


# In[ ]:


# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
predictions


# In[ ]:


print(accuracy_score(y_validation, predictions))


# In[ ]:


print(confusion_matrix(y_validation, predictions))


# # Model saving

# In[ ]:


from pickle import dump
from pickle import load


# In[ ]:


import pickle 
pickle.dump(model, open('Rforest_model.pkl', 'wb')) 


# In[ ]:


loaded_model = load(open('Rforest_model.pkl', 'rb'))


# In[ ]:


result = loaded_model.score(rescaledValidationX, y_validation)
print(result)


# In[ ]:


y_validation


# In[ ]:


st.write("The Prediction value is ",prediction[0])
st.subheader("Predicted Result")
st.write(" 'Yes' the customer will churn " if prediction==1 else " The customer will NOT churn ")


# ## =========================================================================
