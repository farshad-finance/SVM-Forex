#author farshad noravesh
#email: bmmturbo@icloud.com
#book chapter: "machine learning in financial markets"
import pandas as pd
import sys
#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)
#print str(sys.argv[1])
df = pd.read_csv ('supervisorH1.csv')
#print(df)
#df.head()
#df[df.GBPUSDOrderType=='sell'].head()
from matplotlib import pyplot as plt
dfDoNothing=df[df.GBPUSDOrderType=='doNothing']
dfSell=df[df.GBPUSDOrderType=='sell']
#dfSell.head()
plt.xlabel('AUDUSDGBPUSDSpread')
plt.ylabel('EURUSDGBPUSDSpread')
plt.scatter(dfDoNothing['AUDUSDGBPUSDSpread'],dfDoNothing['EURUSDGBPUSDSpread'],color='green',marker='+')
plt.scatter(dfSell['AUDUSDGBPUSDSpread'],dfSell['EURUSDGBPUSDSpread'],color='red',marker='.')
plt.xlabel('AUDUSDGBPUSDSpread')
plt.ylabel('USDCADGBPUSDSpread')
plt.scatter(dfDoNothing['AUDUSDGBPUSDSpread'],dfDoNothing['USDCADGBPUSDSpread'],color='green',marker='+')
plt.scatter(dfSell['AUDUSDGBPUSDSpread'],dfSell['USDCADGBPUSDSpread'],color='red',marker='.')
from sklearn.model_selection import train_test_split
X=df.drop(['GBPUSDOrderType'],axis='columns')
y=df['GBPUSDOrderType']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
len(X_train)
len(X_test)
from sklearn.svm import SVC
model=SVC(kernel='rbf',C=0.65,gamma='auto')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
#print(model.predict([[sys.argv[1],sys.argv[2],sys.argv[3]]]))
print(model.predict([[-0.3003939875999999,-0.2620376019499999,-0.2692853177000001]]))




