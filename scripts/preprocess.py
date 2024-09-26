import pandas as pd

data = pd.read_csv('data/en_train_cleaned.csv')
x_h = data[data['label']==0]
x_m = data[data['label']==1]

pd_m = pd.merge(x_m,x_h, on=['id','question','source'],how='inner',suffixes=('_m','_h'))
pd_m = pd_m[['id','question','source','answer_h','answer_m']]
pd_m = pd_m.rename(columns = {'answer_h':'human','answer_m':'ChatGPT'})
pd_m.to_csv('data/HC3cleaned_LLMs.csv',index=False)