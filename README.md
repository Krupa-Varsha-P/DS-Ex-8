# Ex-08-Data-Visualization-
# AIM
To Perform Data Visualization on a complex dataset and save the data to a file.

# Explanation
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# ALGORITHM
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature generation and selection techniques to all the features of the data set

## STEP 4
Apply data visualization techniques to identify the patterns of the data.

# CODE
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
uploaded = files.upload()
df=pd.read_csv("SuperStore.csv",encoding='unicode_escape')
df
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/aad2b1e0-1d06-4a37-af88-44326a96fc6b)

```
df.isnull().sum()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/b1c4f305-1fb0-4f69-b003-f2288eca6408)

```
df.drop('Row ID',axis=1,inplace=True)
df.drop('Order ID',axis=1,inplace=True)
df.drop('Customer ID',axis=1,inplace=True)
df.drop('Customer Name',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Postal Code',axis=1,inplace=True)
df.drop('Product ID',axis=1,inplace=True)
df.drop('Product Name',axis=1,inplace=True)
df.drop('Order Date',axis=1,inplace=True)
df.drop('Ship Date',axis=1,inplace=True)
print("Updated dataset")
df
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/abbe0ae3-c8fb-4c96-9f88-d26b6dcf20a8)

```
#detecting and removing outliers in current numeric data
plt.figure(figsize=(8,8))
plt.title("Data with outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/6e840e4e-af16-43a4-acad-2cd6445ad1e2)

```
plt.figure(figsize=(8,8))
cols = ['Sales','Discount','Profit']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/401e2dc0-f794-4545-a250-a6f95bd61d95)

## Which Segment has Highest sales?
```
sns.lineplot(x="Segment",y="Sales",data=df,marker='o')
plt.title("Segment vs Sales")
plt.xticks(rotation = 90)
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/a60f5966-a3b4-41da-9d1e-ee70869f54ac)

```
sns.barplot(x="Segment",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/796b3dce-213e-4e05-9dc8-3607d2301320)


## Which City has Highest profit?
```
df.shape
df1 = df[(df.Profit >= 60)]
df1.shape
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/00231b85-0416-418d-8989-2110210e196f)

```
plt.figure(figsize=(30,8))
states=df1.loc[:,["City","Profit"]]
states=states.groupby(by=["City"]).sum().sort_values(by="Profit")
sns.barplot(x=states.index,y="Profit",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("City")
plt.ylabel=("Profit")
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-08/assets/135130074/828646be-91c9-4b2d-9165-4dce050eec06)

## Which ship mode is profitable?
```
sns.barplot(x="Ship Mode",y="Profit",data=df)
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/fe12656f-c63f-4880-96ea-ce3543651e63)

```
sns.lineplot(x="Ship Mode",y="Profit",data=df)
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/978ab79c-1f87-41b7-8f81-bc47a2ff2014)

```
sns.violinplot(x="Profit",y="Ship Mode",data=df)
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/0e13654c-c336-473d-9482-7afcff11e46b)

```
sns.pointplot(x=df["Profit"],y=df["Ship Mode"])
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/3d9052b2-2b87-4e71-b6a5-841dea21581c)


## Sales of the product based on region.
```
states=df.loc[:,["Region","Sales"]]
states=states.groupby(by=["Region"]).sum().sort_values(by="Sales")
sns.barplot(x=states.index,y="Sales",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("Region")
plt.ylabel=("Sales")
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/d77f4a83-7af2-4525-b3e0-61aa4a646769)

```
df.groupby(['Region']).sum().plot(kind='pie', y='Sales',figsize=(6,9),pctdistance=1.7,labeldistance=1.2)
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/81d6a860-b8a8-4e16-a69e-f651b5492da1)


## Find the relation between sales and profit.
```
df["Sales"].corr(df["Profit"])
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/ad560071-83a8-4c99-8d86-1b7738238ccb)

```
df_corr = df.copy()
df_corr = df_corr[["Sales","Profit"]]
df_corr.corr()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/460fa3bd-847e-436f-bf1e-1972e7cf2bd2)

```
sns.pairplot(df_corr, kind="scatter")
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/a107184d-32f4-4aa8-9545-0f7bc0b25ee8)


## Heatmap
```
df4=df.copy()

#encoding
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder
oe=OrdinalEncoder()

df4["Ship Mode"]=oe.fit_transform(df[["Ship Mode"]])
df4["Segment"]=oe.fit_transform(df[["Segment"]])
df4["City"]=le.fit_transform(df[["City"]])
df4["State"]=le.fit_transform(df[["State"]])
df4['Region'] = oe.fit_transform(df[['Region']])
df4["Category"]=oe.fit_transform(df[["Category"]])
df4["Sub-Category"]=le.fit_transform(df[["Sub-Category"]])

#scaling
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df5=pd.DataFrame(sc.fit_transform(df4),columns=['Ship Mode', 'Segment', 'City', 'State','Region',
                                               'Category','Sub-Category','Sales','Quantity','Discount','Profit'])

#Heatmap
plt.subplots(figsize=(12,7))
sns.heatmap(df5.corr(),cmap="PuBu",annot=True)
plt.show()

```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/7fc417f8-e420-4079-be80-7eaf2169e3a8)

## Find the relation between sales and profit based on the following category.

### Segment
```
grouped_data = df.groupby('Segment')[['Sales', 'Profit']].mean()
# Create a bar chart of the grouped data
fig, ax = plt.subplots()
ax.bar(grouped_data.index, grouped_data['Sales'], label='Sales')
ax.bar(grouped_data.index, grouped_data['Profit'], bottom=grouped_data['Sales'], label='Profit')
ax.set_xlabel('Segment')
ax.set_ylabel('Value')
ax.legend()
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/721d49ce-8d42-4a70-8cfa-4d55eaf347b2)

### City
```
grouped_data = df.groupby('City')[['Sales', 'Profit']].mean()
# Create a bar chart of the grouped data
fig, ax = plt.subplots()
ax.bar(grouped_data.index, grouped_data['Sales'], label='Sales')
ax.bar(grouped_data.index, grouped_data['Profit'], bottom=grouped_data['Sales'], label='Profit')
ax.set_xlabel('City')
ax.set_ylabel('Value')
ax.legend()
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/d4373c67-d0f3-486e-92c7-41f1b42d5a23)


### States
```
grouped_data = df.groupby('State')[['Sales', 'Profit']].mean()
# Create a bar chart of the grouped data
fig, ax = plt.subplots()
ax.bar(grouped_data.index, grouped_data['Sales'], label='Sales')
ax.bar(grouped_data.index, grouped_data['Profit'], bottom=grouped_data['Sales'], label='Profit')
ax.set_xlabel('State')
ax.set_ylabel('Value')
ax.legend()
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/4a378e29-8fe6-468e-99b0-12f594a774e4)

### Segment and Ship Mode
```
grouped_data = df.groupby(['Segment', 'Ship Mode'])[['Sales', 'Profit']].mean()
pivot_data = grouped_data.reset_index().pivot(index='Segment', columns='Ship Mode', values=['Sales', 'Profit'])
# Create a bar chart of the grouped data
fig, ax = plt.subplots()
pivot_data.plot(kind='bar', ax=ax)
ax.set_xlabel('Segment')
ax.set_ylabel('Value')
plt.legend(title='Ship Mode')
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/ddcfe2d7-62e5-46c5-b597-a4e5d0093216)


### Segment, Ship mode and Region
```
grouped_data = df.groupby(['Segment', 'Ship Mode','Region'])[['Sales', 'Profit']].mean()
pivot_data = grouped_data.reset_index().pivot(index=['Segment', 'Ship Mode'], columns='Region', values=['Sales', 'Profit'])
sns.set_style("whitegrid")
sns.set_palette("Set1")
pivot_data.plot(kind='bar', stacked=True, figsize=(10, 5))
plt.xlabel('Segment-Ship Mode')
plt.ylabel('Value')
plt.legend(title='Region')
plt.show()
```
![image](https://github.com/Anuayshh/expt8ds/assets/127651217/e8cca182-a525-4104-9214-914aea829178)


# RESULT:
Thus, Data Visualization is performed on the given dataset and save the data to a file.
