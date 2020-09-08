import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

fifa=pd.read_csv('data.csv')

fifa.head()

fifa.columns

fifa.dtypes

fifa.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)

fifa.isnull().sum()

missing_height = fifa[fifa['Height'].isnull()].index.tolist()
missing_weight = fifa[fifa['Weight'].isnull()].index.tolist()
if missing_height == missing_weight:
    print('They are same')
else:
    print('They are different')

fifa.drop(fifa.index[missing_height],inplace =True)

with pd.option_context('display.max_rows',None,'display.max_columns',None):
    print(fifa.isnull().sum())

fifa.drop(['Loaned From','Release Clause','Joined'],axis=1,inplace=True)

print(fifa['Nationality'].value_counts().head(5))
print(fifa['Club'].value_counts().head(5))


def value_to_int(fifa_value):
    try:
        value=float(fifa_value[1:-1])
        suffix=fifa_value[-1:]
        if suffix=='M':
            value=value*1000000
        elif suffix=='K':
            value=value*1000
    except ValueError:
        value=0
    return value

fifa['Value']=fifa['Value'].apply(value_to_int)
fifa['Wage']=fifa['Wage'].apply(value_to_int)

    

fifa.head()


print('Most valued player : '+str(fifa.loc[fifa['Value'].idxmax()][1]))
print('Highest earner : '+str(fifa.loc[fifa['Wage'].idxmax()][1]))

sns.jointplot(x=fifa['Age'],y=fifa['Potential'])

sns.lmplot(data = fifa, x = 'Age', y = 'SprintSpeed')

fifa.columns

drop_cols = fifa.columns[28:54]
fifa = fifa.drop(drop_cols, axis = 1)


fifa = fifa.drop(['ID','Jersey Number','Special','Body Type','Weight','Height','Contract Valid Until','Wage','Value','Name','Club'], axis = 1)


fifa = fifa.dropna()
fifa.head()

fifa = fifa.drop('Position',axis=1)

fifa = fifa.drop('Work Rate',axis=1)

def face_to_num(fifa):
    if (fifa['Real Face'] == 'Yes'):
        return 1
    else:
        return 0
    
def right_footed(fifa):
    if (fifa['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

nat_counts = fifa.Nationality.value_counts()
nat_list = nat_counts[nat_counts > 250].index.tolist()


def major_nation(df):
    if (df.Nationality in nat_list):
        return 1
    else:
        return 0

fifa1 = fifa.copy()    
fifa1['Real_Face'] = fifa1.apply(face_to_num, axis=1)
fifa1['Right_Foot'] = fifa1.apply(right_footed, axis=1)
fifa1['Major_Nation'] = fifa1.apply(major_nation,axis = 1)
fifa1 = fifa1.drop(['Preferred Foot','Real Face', 'Nationality'], axis = 1)

fifa1 = fifa1.drop(['LS','ST','RS','LW','LF','CF'], axis = 1)

y=fifa1['Overall']
X=fifa1.drop(['Overall'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train.shape


X_test.shape

y_train.shape

y_test.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)



from sklearn.metrics import r2_score, mean_squared_error
print('r2 score: '+str(r2_score(y_test, predictions)))
print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))

print('Train Score: ', model.score(X_train,y_train))  
print('Test Score: ', model.score(X_test, y_test))  

#Visualising the results
plt.figure(figsize=(18,10))
sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Linear Prediction of Player Rating")
plt.show()

