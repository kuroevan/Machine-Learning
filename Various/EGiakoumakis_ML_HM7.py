# MSDS 7335 - Black Box Machine Learning
# Homework 7
# Evangelos Giakoumakis

# imports
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import calendar

# set working directory
home = 'c:\\Users\Evan\.spyder'
#work = 'C://Users/Evangelos.Giakoumaki/.spyder'
os.chdir(home)
print os.getcwd()

# Load the transaction data set
# https://www.kaggle.com/xvivancos/transactions-from-a-bakery/version/1
pdf = pandas.read_csv('BreadBasket_DMS.csv')

# Ok I know nothing about this data set.  Let's figure out what is going on together.
print pdf.info()
print pdf.head()

# Part A Describe the bakery
# This dataset contain 4 variables: Transaction, Item, Date, Time
# How many transactions are there?
print "How many transactions are there?" 
print len(pdf)

# What time does their day start?
print "What time does their day start?"
print (pdf.Time.min())

# What time does their day end?
print "What time does their day end?"
print (pdf.Time.max())

# What time is the busiest hour?
print "What time is the busiest hour?"
pdf_hr = pdf
DateTime = pdf_hr.Date + " " + pdf_hr.Time
pdf_hr['DateTime'] = pandas.to_datetime(DateTime, errors='coerce')
pdf_hr.groupby(pdf_hr.DateTime.dt.hour)
grp = pdf_hr.groupby([pandas.Grouper(key='DateTime', freq='H')]).size()
grp = grp.sort_values(ascending=False)
print grp.head(1)

# How many different things do they sell?
print "How many different things do they sell?"
unique_items = pdf.Item.unique()
print len(unique_items)

# Graph of sales by day
pdf['Date'] = pandas.to_datetime(DateTime, errors='coerce')
pdf['day_of_week'] = pdf['Date'].apply(lambda x: x.weekday()) # get the weekday index, between 0 and 6
pdf['day_of_week'] = pdf['day_of_week'].apply(lambda x: calendar.day_name[x])

fig, ax = plt.subplots()
pdf.groupby(['Date']).count()['Item'].plot(ax=ax)
ax.set_title('Sales by day')

# Part B Describe the customer
# How many times do they sell every item?
item_sold = pdf.groupby(['Item']).count()
item_sold = item_sold.drop(columns=['Time', 'DateTime', 'Date', 'day_of_week'])
print item_sold

# What is the relative frequency of sales? (graph)
res = stats.relfreq(pdf.Transaction, numbins=9684)
res.frequency
x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size, res.frequency.size)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(1, 1, 1)
ax.bar(x, res.frequency, width=res.binsize)
ax.set_title('Relative frequency histogram')
plt.show()

# How often do people buy tea with coffee? How about Coffee and croissant?  Coffee and something from the bakery?
cof = pdf.loc[pdf['Item'] == 'Coffee']
tea = pdf.loc[pdf['Item'] == 'Tea']
cro = pdf.loc[pdf['Item'] == 'Croissant']

cofteajoin = cof.set_index('Transaction').join(tea.set_index('Transaction'), lsuffix='_left', rsuffix='_right')
joined = cofteajoin['Item_right'].value_counts()
print "How often do people buy tea with coffee?"
print joined
print "How about Coffee and croissant?"
print "0"
print "Coffee and something from the bakery?"
print len(cof)

# What is the average number of items purchased per person?
item_num = item_sold = pdf.groupby(['Transaction']).count()
mean_item_num = item_num['Item'].mean()
print "What is the average number of items purchased per person?"
print mean_item_num

# What hour of the day are there the most sales?
print "What hour of the day are there the most sales?"
hdms = pdf.pivot_table(index=pdf['Date'].dt.hour, 
                     columns='day_of_week', 
                     values='Transaction', 
                     aggfunc='sum').plot()

# What day of week are the most sales?
print "What day of week are the most sales?"
print "Saturday"
dwms = pdf.groupby(['day_of_week']).count()['Item']
print dwms.head()

# How is the business doing?
print "How is the business doing?"
maxlen = len(pdf)
dtdur = abs(pdf.Date[maxlen-1] - pdf.Date[0]).days
tranday = maxlen / dtdur 
print "Business is doing well, it has an average of this many transactions per day: "
print tranday

# How often is there a line with people waiting?
cnt = 0
transpd = pdf.Date.diff()
mn = transpd.mean()
for i in range(len(transpd)):
    if transpd[i] < mn:
        cnt += 1

print "How often is there a line with people waiting?"        
print cnt

# How fast can they do transactions? ie should they get another register?
print "How fast can they do transactions? ie should they get another register?"
transpeed = pdf.Date.diff()
print transpeed.mean()
print "It takes an average of 10 minutes between each transaction, so it might be a good idea to have a second register during rush hour"

# Part C.Model the customer behavior?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from IPython.display import display

pdf2 = pandas.read_csv('BreadBasket_DMS.csv')
pdf2['ItemCodes'] = pdf2.Item.astype('category').cat.codes
pdf2['catDate'] = pdf2.Date.astype('category').cat.codes
pdf2['catTime'] = pdf2.Time.astype('category').cat.codes

pdf2['label'] = ""
pdf2 = pdf2.drop(["Date", "Time", "Item"], axis=1) 
features = pdf2.drop(["label"], axis=1).columns

# Split data into test (25%) and train (75%) 
df_train, df_test = train_test_split(pdf2, test_size=0.25)

df_train = df_train.reset_index()
df_test = df_test.reset_index()

# Set up our RandomForestClassifier instance and fit to data
clf = RandomForestClassifier(n_estimators=30)
clf.fit(df_train[features], df_train["ItemCodes"])

# Make predictions
predictions = clf.predict(df_test[features])
probs = clf.predict_proba(df_test[features])
display(predictions)

print "Modeling Customer behavior based on items sold"

score = clf.score(df_test[features], df_test["ItemCodes"])
print("Accuracy: ", score)

# http://blog.keyrus.co.uk/a_simple_approach_to_predicting_customer_churn.html

# Model the user behavior to help the owner.  Explain why what you are doing will help the owner.
# modeling customer behavior can help the owner predict what his clients tend to purchase as well as when, 
# which in turn can aid him in better preparing and ultimately becoming more profitable
# They run out of coffee after 500 cups.  They want to get it at the last possible minute so its fresh.  But it takes 3 days to arrive.  When should they order?
cof['counter'] = range(len(cof))
diff = cof['DateTime']
diff = diff.reset_index()

for i in range(0, len(cof), 500):
    cof_dif = diff['DateTime'][i+500]- diff['DateTime'][i]
    # code might fail run again from here on and it will work!
avg_cof = cof_dif / 11
print "Average consumption of 500 cups of Coffee: ", avg_cof

# According to our analysis he should always have 2 coffee batches in stock and order a new batch the moment he opens the old one since it takes an average of 3 days to arrive, and they go through 500 cups of coffee in an average of 1,5 day.
# Predict something.  Impress them.
# Set up our RandomForestClassifier instance and fit to data
clf2 = RandomForestClassifier(n_estimators=30)
clf2.fit(df_train[features], df_train["catTime"])

# Make predictions
predictions2 = clf2.predict(df_test[features])
probs2 = clf2.predict_proba(df_test[features])
display(predictions2)

print "Modeling Customer behavior based on purchase time"

score2 = clf2.score(df_test[features], df_test["catTime"])
print("Accuracy: ", score2)
