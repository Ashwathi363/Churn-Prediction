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
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix,roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
from IPython import get_ipython


# In[2]:


ipython = get_ipython()
ipython.magic("timeit abs(-42)")


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

# # **Exploratory Data Analysis Of The Data Set**

# ### Analyzing What The Dependent Variable Said To Us i.e 'CHURN'.

# In[20]:


#convert string values(yes and no) of churn column to 1 and 0
data.loc[data.churn=='no', 'churn']=0
data.loc[data.churn=='yes', 'churn']=1
#convert to integer
data['churn']=data['churn'].astype('int32')


# In[21]:


#Printing the unique value inside "churn" column
data["churn"].unique()


# In[22]:


#Printing the count of true and false in 'churn' feature
print(data.churn.value_counts())


# In[23]:


100*data['churn'].value_counts()/len(data['churn'])


# In[24]:


#To get the pie chart to analyze churn
data['churn'].value_counts().plot.pie(explode=[0.05,0.05], autopct='%1.1f%%',  startangle=90,shadow=True, figsize=(8,8))
plt.title('Pie Chart for Churn')
plt.show()


# In[25]:


#let's see churn by using countplot
sns.countplot(x=data.churn)


# In[26]:


sns.boxplot(x=data['churn'])


# ***After analyzing the churn column, we had little to say like almost 15% of customers have churned. let's see what other features say to us and what relation we get after correlated with churn***

# ### Analyzing State Column

# In[27]:


#printing the unique value of sate column
data['state'].nunique()


# In[28]:


#Comparison churn with state by using countplot (barchart)
sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
ax = sns.countplot(x='state', hue="churn", data=data)
plt.show()


# In[29]:


plt.rcParams['figure.figsize'] = (12, 7)
color = plt.cm.copper(np.linspace(0, 0.5, 20))
((data.groupby(['state'])['churn'].mean())*100).sort_values(ascending = False).head(6).plot.bar(color = ['violet','indigo','b','g','y','orange','r'])
plt.title(" State with most churn percentage", fontsize = 20)
plt.xlabel('state', fontsize = 15)
plt.ylabel('percentage', fontsize = 15)
plt.show()


# In[30]:


#calculate State vs Churn percentage
State_data = pd.crosstab(data["state"],data["churn"])
State_data['Percentage_Churn'] = State_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(State_data)


# In[31]:


#show the most churn state of top 10 by ascending the above list
data.groupby(['state'])['churn'].mean().sort_values(ascending = False).head(10)


# ***There is 51 unique state present who have different churn rate.*** 
# 
# ***From the above analysis CA, NJ, WA, TX, MT, MD, NV, ME, KS, OK are the ones who have a higher churn rate of more than 21.***
# 
#  ***The reason for this churn rate from a particular state may be due to the low coverage of the cellular network.***

# ### Analyzing "Area Code" column

# In[32]:


#calculate Area code vs Churn percentage
Area_code_data = pd.crosstab(data["area.code"],data["churn"])
Area_code_data['Percentage_Churn'] = Area_code_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Area_code_data)


# In[33]:


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

# In[34]:


data1= data.drop(columns=['state','area.code'],axis=1)


# In[35]:


data1.head()


# ### Analyzing "Account length" column

# In[36]:


#Separating churn and non churn customers
churn_df     = data1[data1["churn"] == bool(True)]
not_churn_df = data1[data1["churn"] == bool(False)]


# In[37]:


#Account length vs Churn
sns.distplot(data1['account.length'])


# In[38]:


#comparison of churned account length and not churned account length 
sns.distplot(data1['account.length'],color = 'purple',label="All")
sns.distplot(churn_df['account.length'],color = "red",hist=False,label="Churned")
sns.distplot(not_churn_df['account.length'],color = 'green',hist= False,label="Not churned")
plt.legend()


# In[39]:


fig = plt.figure(figsize =(10, 8)) 
data1.boxplot(column='account.length', by='churn')
fig.suptitle('account.length', fontsize=14, fontweight='bold')
plt.show()


# ***After analyzing various aspects of the "account length" column we didn't found any useful relation to churn. so we aren't able to build any connection to the churn as of now. let's see what other features say about the churn.***

# ### Analyzing "International Plan" column

# In[40]:


#Show count value of 'yes','no'
data1['intl.plan'].value_counts()


# In[41]:


#Show the unique data of "International plan"
data1["intl.plan"].unique()


# In[42]:


#Calculate the International Plan vs Churn percentage 
International_plan_data = pd.crosstab(data1["intl.plan"],data1["churn"])
International_plan_data['Percentage Churn'] = International_plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(International_plan_data)


# In[43]:


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

# In[44]:


#show the unique value of the "Voice mail plan" column
data1["voice.plan"].unique()


# In[45]:


#Calculate the Voice Mail Plan vs Churn percentage
Voice_mail_plan_data = pd.crosstab(data1["voice.plan"], data1["churn"])
Voice_mail_plan_data['Percentage Churn'] = Voice_mail_plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Voice_mail_plan_data)


# In[46]:


#Analysing by using countplot
sns.countplot(x='voice.plan',hue="churn",data = data1)


# ***As we can see there is are no clear relation between voice mail plan and churn so we can't clearly say anything so let's move to the next voice mail feature i.e number of voice mail, let's see what it gives to us.***

# ### Analyzing "Voice messages" column

# In[47]:


#show the data of 'voice messages' 
data1['voice.messages'].unique()


# In[48]:


#Printing the data of 'voice messages'
data1['voice.messages'].value_counts()


# In[49]:


#Show the details of 'voice messages' data
data1['voice.messages'].describe()


# In[50]:


#Analysing by using displot diagram
sns.distplot(data1['voice.messages'])


# In[51]:


#Analysing by using boxplot diagram between 'voice messages' and 'churn'
fig = plt.figure(figsize =(10, 8)) 
data1.boxplot(column='voice.messages', by='churn')
fig.suptitle('voice message', fontsize=14, fontweight='bold')
plt.show()


# ***After analyzing the above voice message feature data we get an insight that when there are more than 20 voice-mail messages then  there is a churn***
# 
# ***For that, we need to improve the voice message quality.***

# ### Analyzing "Customer calls" column

# In[52]:


#Printing the data of customer service calls 
data1['customer.calls'].value_counts()


# In[53]:


#Calculating the Customer calls vs Churn percentage
Customer_service_calls_data = pd.crosstab(data1['customer.calls'],data1["churn"])
Customer_service_calls_data['Percentage_Churn'] = Customer_service_calls_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Customer_service_calls_data)


# In[54]:


#Analysing using countplot
sns.countplot(x='customer.calls',hue="churn",data = data1)


# In[55]:


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

# In[56]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['day.calls'].mean())


# In[57]:


sns.boxplot(x='churn', y='day.calls', data=data1)


# In[58]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['day.mins'].mean())


# In[59]:


sns.boxplot(x='churn', y='day.mins', data=data1)


# In[60]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['day.charge'].mean())


# In[61]:


sns.boxplot(x='churn', y='day.charge', data=data1)


# In[62]:


#show the relation using scatter plot
sns.scatterplot(x="day.mins", y="day.charge", hue="churn", data=data1, palette='hls')


# In[63]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['eve.calls'].mean())


# In[64]:


sns.boxplot(x='churn', y='eve.calls', data=data1)


# In[65]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['eve.mins'].mean())


# In[66]:


sns.boxplot(x='churn', y='eve.mins', data=data1)


# In[67]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['eve.charge'].mean())


# In[68]:


sns.boxplot(x='churn', y='eve.charge', data=data1)


# In[69]:


#show the relation using scatter plot
sns.scatterplot(x="eve.mins", y="eve.charge", hue="churn", data=data1, palette='hls')


# In[70]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['night.calls'].mean())


# In[71]:


sns.boxplot(x='churn', y='night.calls', data=data1)


# In[72]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['night.charge'].mean())


# In[73]:


sns.boxplot(x='churn', y='night.charge', data=data1)


# In[74]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['night.mins'].mean())


# In[75]:


sns.boxplot(x='churn', y='night.mins', data=data1)


# In[76]:


#show the relation using scatter plot
sns.scatterplot(x="night.mins", y="night.charge", hue="churn", data=data1, palette='hls')


# In[77]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['intl.mins'].mean())


# In[78]:


sns.boxplot(x='churn', y='intl.mins', data=data1)


# In[79]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['intl.calls'].mean())


# In[80]:


sns.boxplot(x='churn', y='intl.calls', data=data1)


# In[81]:


#Print the mean value of churned and not churned customer 
print(data1.groupby(["churn"])['intl.charge'].mean())


# In[82]:


sns.boxplot(x='churn', y='intl.charge', data=data1)


# In[83]:


#show the relation using scatter plot
sns.scatterplot(x="intl.mins", y="intl.charge", hue="churn", data=data1, palette='hls')


# In[84]:


#Deriving a relation between overall call charge and overall call minutes   
day_charge_perm = data1['day.charge'].mean()/data1['day.mins'].mean()
eve_charge_perm = data1['eve.charge'].mean()/data1['eve.mins'].mean()
night_charge_perm = data1['night.charge'].mean()/data1['night.mins'].mean()
int_charge_perm= data1['intl.charge'].mean()/data1['intl.mins'].mean()


# In[85]:


print([day_charge_perm,eve_charge_perm,night_charge_perm,int_charge_perm])


# In[86]:


sns.barplot(x=['day','Eve','night','intl'],y=[day_charge_perm,eve_charge_perm,night_charge_perm,int_charge_perm])


# 
# ***After analyzing the above dataset we have noticed that total day/night/eve minutes/call/charges are not put any kind of cause for churn rate. But international call charges are high as compare to others it's an obvious thing but that may be a cause for international plan customers to churn out.***

# ## Graphical Analysis

# ### UNIVARIATE ANALYSIS
# 
# In Univariate Analysis we analyze data over a single column from the numerical dataset, for this we use 3 types of plot which are **box plot, strip plot, dis plot.** 
# 
# 

# In[87]:


df1=data1.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.displot(data=df1, x=column, hue='churn')
plt.show()


# In[88]:


#Printing boxplot for each numerical column present in the data set
df1=data1.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=df1, x=column, hue='churn')
plt.show()


# In[ ]:





# In[89]:


#Printing strip plot for each numerical column present in the data set
df1=data.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.stripplot(data=df1, x=column, hue='churn')
plt.show()


# ### converting categorical variables into dummy variables

# In[90]:


data1_dummies = pd.get_dummies(data1)
data1_dummies.head()


# ### Multivariate Analysis
# In Multivariate Analysis we analyze data by taking more than two columns into consideration from a dataset,for this we using **correlation plot,correlation matrix, correletaion heatmap, pair plot**

# In[91]:


#build correlation of all predictors with churn
plt.figure(figsize=(20,8))
data1_dummies.corr()['churn'].sort_values(ascending=False).plot(kind='bar')


# In[92]:


plt.figure(figsize=(17,8))
sns.heatmap(data1_dummies.corr(), cmap="Paired")


# In[93]:


## plot the Correlation matrix
plt.figure(figsize=(17,8))
correlation=data1_dummies.corr()
sns.heatmap(abs(correlation), annot=True, cmap='coolwarm')


# # Finding Outliers in Data

# In[94]:


def find_outliers_IQR(data):
   q1=data.quantile(0.25)
   q3=data.quantile(0.75)
   IQR=q3-q1
   outliers = data[((data<(q1-1.5*IQR)) | (data>(q3+1.5*IQR)))]
   return outliers


# In[95]:


outliers=find_outliers_IQR(data1)


# In[96]:


outliers.head(20)


# In[97]:


outliers.info()


# ## Dropping categorical data from outliers dataframe

# In[98]:


outliers=outliers.drop(columns=['voice.plan','intl.plan','customer.calls','churn'],axis=1)


# In[99]:


outliers.head()


# In[100]:


outliers.info()


# In[101]:


outlier_data=outliers[outliers.values>=0]
outlier_data


# In[102]:


outlier_data.info()


# In[103]:


outlier_data['churn']=data1['churn']


# In[104]:


outlier_data


# In[105]:


outlier_data1=outlier_data[~outlier_data.index.duplicated(keep='first')]
outlier_data1


# In[106]:


outlier_data1.info()


# In[107]:


print(outlier_data1.churn.value_counts())
outlier_data1['churn'].value_counts().plot(kind="pie",autopct="%1.2f%%",figsize=(5,5))


# ### There are 391 customers who are outliers but they are not churns customers
# ### And 87 customers are outliers but they are churns

# In[108]:


data1.boxplot(figsize=(20,10))


# # Removing not churn customer outliers

# In[109]:


nc=outlier_data1.index[outlier_data1['churn']==0]


# In[110]:


ncco=data1.drop(nc)


# In[111]:


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

# In[112]:


ncco.to_csv('updated_churn.csv')


# # Model building

# In[113]:


df2 = pd.read_csv('updated_churn.csv', index_col = 0)

encode = LabelEncoder()
df2['voice.plan'] = encode.fit_transform(df2['voice.plan'])
df2['intl.plan'] = encode.fit_transform(df2['intl.plan'])

scaler = MinMaxScaler()
df2.iloc[:,[0,2,4,6,7,9,10,12,13,15,5,8,11,14]] = scaler.fit_transform(df2.iloc[:,[0,2,4,6,7,9,10,12,13,15,5,8,11,14]])

df2.head()


# In[114]:


df2.shape


# In[115]:


X = df2.drop(labels='churn', axis=1)
y = df2['churn']


# In[116]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, shuffle= True, random_state=27, stratify=y)


# # Oversampling

# In[117]:


#SMOTE Technique
get_ipython().system('pip install imbalanced-learn --upgrade')
from collections import Counter
from imblearn.over_sampling import SMOTE,  ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE
smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_sm)
print('After',counter)


# In[118]:


counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE + Tomek
smtom = SMOTETomek(random_state=139)
X_train_smtom, y_train_smtom = smtom.fit_resample(X_train, y_train)

counter = Counter(y_train_smtom)
print('After',counter)


# In[119]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# # Logistic Regression

# In[120]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[121]:


y_train_pred1_lr = lr.predict(X_train)


# In[122]:


acc_logistic_train1 = accuracy_score(y_train, y_train_pred1_lr)
acc_logistic_train1


# In[123]:


#train data
print(classification_report(y_train, y_train_pred1_lr))


# In[124]:


y_test_pred1_lr = lr.predict(X_test)


# In[125]:


acc_logistic_test1 = accuracy_score(y_test, y_test_pred1_lr)
acc_logistic_test1


# In[126]:


#test data
print(classification_report(y_test, y_test_pred1_lr))


# ### using SMOTE TOMEK

# In[127]:


lr3 = LogisticRegression()      
lr3.fit(X_train_smtom, y_train_smtom)


# In[128]:


y_train_pred1_lr3 = lr3.predict(X_train_smtom)


# In[129]:


acc_logistic_train3 = accuracy_score(y_train_smtom, y_train_pred1_lr3)
acc_logistic_train3


# In[130]:


#train data
print(classification_report(y_train_smtom, y_train_pred1_lr3 ))


# In[131]:


y_test_pred1_lr3 = lr3.predict(X_test)


# In[132]:


acc_logistic_test3 = accuracy_score(y_test, y_test_pred1_lr3)
acc_logistic_test3


# In[133]:


#test data
print(classification_report(y_test,y_test_pred1_lr3))


# # Decision Tree

# In[134]:


dt =  DecisionTreeClassifier()        
dt.fit(X_train, y_train)


# In[135]:


y_train_pred1_dt = dt.predict(X_train)


# In[136]:


acc_Decision_train = accuracy_score(y_train, y_train_pred1_dt)
acc_Decision_train


# In[137]:


# train data
print(classification_report(y_train,y_train_pred1_dt))


# In[138]:


y_test_pred1_dt = dt.predict(X_test)


# In[139]:


acc_Decision_test = accuracy_score(y_test, y_test_pred1_dt)
acc_Decision_test


# In[140]:


#test data
print(classification_report(y_test, y_test_pred1_dt))


# ### using SMOTE TOMEK

# In[141]:


dt3 =  DecisionTreeClassifier()        
dt3.fit(X_train_smtom, y_train_smtom)                


# In[142]:


y_train_pred1_dt3 = dt.predict(X_train_smtom)


# In[143]:


acc_Decision_train3 = accuracy_score(y_train_smtom, y_train_pred1_dt3)
acc_Decision_train3


# In[144]:


# train data
print(classification_report(y_train_smtom,y_train_pred1_dt3))


# In[145]:


y_test_pred1_dt3 = dt.predict(X_test)


# In[146]:


acc_Decision_test3 = accuracy_score(y_test, y_test_pred1_dt3)
acc_Decision_test3


# In[147]:


# test data
print(classification_report(y_test,  y_test_pred1_dt3))


# # Naive Bayes

# In[148]:


nb =  GaussianNB()        
nb.fit(X_train, y_train)


# In[149]:


y_train_pred1_nb = nb.predict(X_train)


# In[150]:


acc_Naive_train = accuracy_score(y_train, y_train_pred1_nb)
acc_Naive_train


# In[151]:


#train data
print(classification_report(y_train,y_train_pred1_nb))


# In[152]:


y_test_pred1_nb = nb.predict(X_test)


# In[153]:


acc_Naive_test = accuracy_score(y_test, y_test_pred1_nb)
acc_Naive_test


# In[154]:


# test data
print(classification_report(y_test, y_test_pred1_nb))


# ### using SMOTE TOMEK

# In[155]:


nb3 =  GaussianNB()        
nb3.fit(X_train_smtom, y_train_smtom)                  #smtom


# In[156]:


y_train_pred1_nb3 = nb3.predict(X_train)


# In[157]:


acc_Naive_train3 = accuracy_score(y_train, y_train_pred1_nb3)
acc_Naive_train3


# In[158]:


#train data
print(classification_report(y_train,y_train_pred1_nb3))


# In[159]:


y_test_pred1_nb3 = nb3.predict(X_test)


# In[160]:


acc_Naive_test3 = accuracy_score(y_test, y_test_pred1_nb3)
acc_Naive_test3


# In[161]:


#test data
print(classification_report(y_test, y_test_pred1_nb3))


# # SVM

# In[162]:


sm =  SVC()        
sm.fit(X_train, y_train)


# In[163]:


y_train_pred1_sm = sm.predict(X_train)


# In[164]:


acc_svm_train = accuracy_score(y_train, y_train_pred1_sm)
acc_svm_train


# In[165]:


#train data
print(classification_report(y_train,y_train_pred1_sm))


# In[166]:


y_test_pred1_sm = sm.predict(X_test)


# In[167]:


acc_svm_test = accuracy_score(y_test, y_test_pred1_sm)
acc_svm_test


# In[168]:


#test data
print(classification_report(y_test,  y_test_pred1_sm))


# ### using SMOTE TOMEK

# In[169]:


sm3 =  SVC()        
sm3.fit(X_train_smtom, y_train_smtom)              #smtom


# In[170]:


y_train_pred1_sm3 = sm3.predict(X_train)


# In[171]:


acc_svm_train3 = accuracy_score(y_train, y_train_pred1_sm3)
acc_svm_train3


# In[172]:


#train data
print(classification_report(y_train, y_train_pred1_sm3))


# In[173]:


y_test_pred1_sm3 = sm3.predict(X_test)


# In[174]:


acc_svm_test3 = accuracy_score(y_test, y_test_pred1_sm3)
acc_svm_test3


# In[175]:


#test data
print(classification_report(y_test,  y_test_pred1_sm3))


# # Random Forest

# In[176]:


rf =  RandomForestClassifier()        
rf.fit(X_train, y_train)


# In[177]:


y_train_pred1_rf = rf.predict(X_train)


# In[178]:


acc_rf_train = accuracy_score(y_train, y_train_pred1_rf)
acc_rf_train


# In[179]:


#train data
print(classification_report(y_train,  y_train_pred1_rf))


# In[180]:


y_test_pred1_rf = rf.predict(X_test)


# In[181]:


acc_rf_test = accuracy_score(y_test, y_test_pred1_rf)
acc_rf_test


# In[182]:


#test data
print(classification_report(y_test,  y_test_pred1_rf))


# ### using SMOTE TOMEK

# In[183]:


rf3 =  RandomForestClassifier()        
rf3.fit(X_train_smtom, y_train_smtom)              #smtom


# In[184]:


y_train_pred1_rf3 = rf3.predict(X_train)


# In[185]:


acc_rf_train3 = accuracy_score(y_train, y_train_pred1_rf3)
acc_rf_train3


# In[186]:


#train data
print(classification_report(y_train,  y_train_pred1_rf3))


# In[187]:


y_test_pred1_rf3 = rf3.predict(X_test)


# In[188]:


acc_rf_test3 = accuracy_score(y_test, y_test_pred1_rf3)
acc_rf_test3


# In[189]:


#test data
print(classification_report(y_test,  y_test_pred1_rf3))


# # Gradient Boosting

# In[190]:


Gb =  GradientBoostingClassifier()        
Gb.fit(X_train, y_train)


# In[191]:


y_train_pred1_Gb = Gb.predict(X_train)


# In[192]:


acc_Gb_train = accuracy_score(y_train, y_train_pred1_Gb)
acc_Gb_train


# In[193]:


#train data
print(classification_report(y_train,  y_train_pred1_Gb))


# In[194]:


y_test_pred1_Gb = Gb.predict(X_test)


# In[195]:


acc_Gb_test = accuracy_score(y_test, y_test_pred1_Gb)
acc_Gb_test


# In[196]:


#test data
print(classification_report(y_test,   y_test_pred1_Gb))


# ### using SMOTE TOMEK

# In[197]:


Gb2 =  GradientBoostingClassifier()        
Gb2.fit(X_train_smtom, y_train_smtom)             #smtom


# In[198]:


y_train_predict = Gb2.predict(X_train_smtom)


# In[199]:


acc_Gb2_train = accuracy_score(y_train_smtom, y_train_predict)
acc_Gb2_train


# In[200]:


#train
print(classification_report(y_train_smtom,   y_train_predict))


# In[201]:


y_test_pred1_Gb2 = Gb2.predict(X_test)


# In[202]:


acc_Gb2_test = accuracy_score(y_test, y_test_pred1_Gb2)
acc_Gb2_test


# In[203]:


#test
print(classification_report(y_test,   y_test_pred1_Gb2))


# In[204]:


plot_confusion_matrix(estimator= Gb2,X=X_test, y_true=y_test,cmap='Blues')
plt.show()


# # Without Sampling

# In[205]:


Acc = pd.DataFrame(data = {'Algorithm': ['Logistic Regression','Decision Tree','Naive Bayes','SVM','Random Forest','Gradient Boosting'],
                           'Train accuracy': [acc_logistic_train1, acc_Decision_train, acc_Naive_train, acc_svm_train, acc_rf_train, acc_Gb_train],
                           'Test accuracy': [acc_logistic_test1, acc_Decision_test,acc_Naive_test,  acc_svm_test, acc_rf_test, acc_Gb_test]})

Acc


# # After Performing Sampling

# In[206]:


New_Acc = pd.DataFrame(data = {'Algorithm': ['Logistic Regression','Decision Tree','Naive Bayes','SVM','Random Forest','Gradient Boosting'],
                           'Train accuracy': [acc_logistic_train3, acc_Decision_train3, acc_Naive_train3, acc_svm_train3, acc_rf_train3, acc_Gb2_train],
                           'Test accuracy': [acc_logistic_test3, acc_Decision_test3, acc_Naive_test3,  acc_svm_test3, acc_rf_test3, acc_Gb2_test]})

New_Acc


# # After comparing accuracy with others we got "gradient boosting" have best train and test accuracy.
