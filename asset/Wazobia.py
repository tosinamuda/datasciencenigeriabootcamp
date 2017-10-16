
# coding: utf-8

# In[207]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[208]:


# read the train data into pandas
read_data = pd.read_csv('train2.csv')


# In[209]:


read_data.describe()


# In[210]:


# this shows the degree of correlation among each variable.
# -1 representing high negative correlation and +1 representing high positive correlation
corr_matrix = read_data.corr()
f, ax = plt.subplots(figsize=(18, 10))
sns.heatmap(corr_matrix, linewidths=2.0, ax=ax, annot=True)
ax.set_title('Correlation Matrix')


# In[211]:


# this removes the 'S/N' variable
new_data = read_data[read_data.columns.difference(['S/N'])]


# In[212]:


new_data.info()


# In[213]:


new_data.iloc[0:5, 0:15]


# In[214]:


new_data.iloc[0:5, 15:]


# In[215]:


# gender: 1==Boy, 0 = Girl
# yes ==1, no === 0
# Pstatus T=1, A=0
# famsize LE3=0, GT3 = 1
# location U=0 , R=1
values = {"activities": {"no":0,"yes":1}, "Gender":{"F":0, "M":1}, "famsup":{"no":0,"yes":1}, 
          "higher":{"no":0,"yes":1}, "internet":{"no":0,"yes":1}, "nursery":{"no":0,"yes":1}, 
          "paid":{"no":0,"yes":1}, "schoolsup":{"no":0,"yes":1}, "Pstatus":{"T":1, "A":0}, 
          "famsize":{"LE3":0, "GT3":1}, "Location":{"U":0, "R":1}}
new_data.replace(values, inplace=True)


# In[216]:


corr_matrix = new_data.corr()
f, ax = plt.subplots(figsize=(18, 10))
sns.heatmap(corr_matrix, linewidths=2.0, ax=ax, annot=True)
ax.set_title('Correlation Matrix')


# In[217]:


# multiply variables with high correlation
new_data["age_high"] = new_data["Age"]* new_data["higher"] 
new_data["fail_high"] = new_data["failures"]* new_data["higher"]
new_data["fail_age"] = new_data["failures"]* new_data["Age"]
new_data["fedu_medu"] = new_data["Medu"]* new_data["Fedu"]
new_data["medu_travel"] = new_data["Medu"]* new_data["traveltime"]
new_data["medu_internet"] = new_data["Medu"]* new_data["internet"]


# In[218]:


corr_matrix = new_data.corr()
f, ax = plt.subplots(figsize=(18, 10))
sns.heatmap(corr_matrix, linewidths=2.0, ax=ax, annot=True)
ax.set_title('Correlation Matrix')


# In[219]:


negative_corr = new_data[['Age','failures', 'absences', 'health', 'schoolsup']]


# In[220]:


positive_corr = new_data[['higher', 'activities', 'Medu', 'studytime', 'Fedu', 'absences' ]]


# In[221]:


# data passed into the Linear regression model
gen_data = new_data[["age_high", "fail_high", "fail_age", "fedu_medu", "higher", "failures", 'activities', 'Medu', 'studytime','absences', 'medu_internet']]


# In[222]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
lm = LinearRegression()


# In[223]:


test_data = pd.read_csv('test.csv')


# In[224]:


values = {"activities": {"no":0,"yes":1}, "Gender":{"F":0, "M":1}, "famsup":{"no":0,"yes":1}, 
          "higher":{"no":0,"yes":1}, "internet":{"no":0,"yes":1}, "nursery":{"no":0,"yes":1}, 
          "paid":{"no":0,"yes":1}, "schoolsup":{"no":0,"yes":1}, "Pstatus":{"T":1, "A":0}, 
          "famsize":{"LE3":0, "GT3":1}, "Location":{"U":0, "R":1}}
test_data.replace(values, inplace=True)


# In[225]:


test_data["age_high"] = test_data["Age"]* test_data["higher"] 
test_data["fail_high"] = test_data["failures"]* test_data["higher"]
test_data["fail_age"] = test_data["failures"]* test_data["Age"]
test_data["fedu_medu"] = test_data["Medu"]* test_data["Fedu"]
test_data["medu_internet"] = test_data["Medu"]* test_data["internet"]


# In[226]:


test_data.head()


# In[227]:


test_again = test_data[["age_high", "fail_high", "fail_age", "fedu_medu", "higher", "failures", 'activities', 'Medu', 'studytime','absences','medu_internet']]


# In[228]:


test_again.head()


# In[229]:


gen_data.head()


# In[230]:


x_train, x_test, y_train, y_test = train_test_split(gen_data, new_data.Score, test_size=0.33, random_state=5)


# In[231]:


lm.fit(x_train, y_train)


# In[235]:


pred_train = lm.predict(x_train)
pred_test = lm.predict(x_test)
# pred_test = lm.predict(test_again)


# In[236]:


h = pd.DataFrame(pred_test)
h['S/N'] = new_data2['S/N']
h.to_csv('pred_test10.csv')


# In[237]:


# x = mean_squared_error(y_test, pred_test)


# In[239]:


# x**0.5

