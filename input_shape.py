import pandas as pd

df=pd.read_csv("AMD_indicators.csv")

series = df[['Close','H-L','7 DAYS STD DEV','RSI','MACD','Volume']] # Picking the series with high correlation

train_start = 5000
train_end = 9000
train_data = series.loc[train_start:train_end-1]

val_start = 9000
val_end = 10000
val_data = series.loc[val_start:val_end-1]

test_start = 10000
test_end = 10254
test_data = series.loc[test_start:test_end-1]

windows_size=20

print(train_data.shape,val_data.shape,test_data.shape)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train = sc.fit_transform(train_data)
val = sc.transform(val_data)
    
def making_train_val(train_data,val_data,windows_size):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
     
    # Loop for training data
    for i in range(windows_size,train.shape[0]):
        X_train.append(train[i-windows_size:i])
        y_train.append(train[i][0])
    X_train,y_train = np.array(X_train),np.array(y_train)
  
    # Loop for validation data
    for i in range(windows_size,val.shape[0]):
        X_val.append(val[i-windows_size:i])
        y_val.append(val[i][0])
    X_val,y_val = np.array(X_val),np.array(y_val)
    
    return X_train,y_train,X_val,y_val

X_train,y_train,X_val,y_val=making_train_val(train_data,val_data,windows_size)