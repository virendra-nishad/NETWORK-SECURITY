import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('big.csv', sep=",", header=None)       #loading data file
feature_name=list(data.columns.values)

data[1] = data[1].astype('category')
data[2] = data[2].astype('category')
data[3] = data[3].astype('category')

cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)


#0:normal
#1:dos
#2:r2l
#3:u2r
#4:probe

attack = {'normal.':0, 'back.':1, 'land.':1, 'neptune.':1, 'pod.':1, 'teardrop.':1, 'smurf.':1, 'ftp_write.':2, 'guess_passwd.':2, 'imap.':2, 'multihop.':2, 'phf.':2, 'spy.':2, 'warezclient.':2, 'warezmaster.':2, 'buffer_overflow.':3, 'loadmodule.':3, 'perl.':3, 'rootkit.':3, 'ipsweep.':4, 'nmap.':4, 'portsweep.':4, 'satan.':4}

data[41] = [attack[t] for t in data[41]] 

#print(data)

data=data.values   
X = data[:,0:41]     #separating labels and features          
y = data[:,41]

#print(y)
#print(X.shape)
#print(y.shape)

train, test = train_test_split(data, test_size=0.25)
                        
df = pd.DataFrame(train)                                 #training data file
df.to_csv("train_s.csv",index=False, header= False)
                      
df = pd.DataFrame(test)                                  #training data file
df.to_csv("test_s.csv",index=False, header= False)

