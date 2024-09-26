import pandas as pd
from groq import Groq
import random
from time import sleep
from tqdm import tqdm
import json
import os
import nltk
import re
import arparse

data_length = 2200

client = Groq(
    api_key=os.environ.get('GROQ_API_KEY')
)

pd_x=pd.read_csv('data/HC3cleaned_LLMs.csv').loc[:data_length,:]

try_limit = 5
sleep_time = 2

for i in tqdm(range(data_length)):
    text = pd_x.loc[i,'human']
    sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences

    con_x = ''
    for j in range(len(sent_text)):
        con_x += f'{j}th sentence: {sent_text[j]}'+'\n'
    idx_list = random.sample(range(len(sent_text)),len(sent_text)//2)
    idx_list.sort()
    cur_prompt = f'''Please paraphrase sentence numbers {idx_list} in given written texts.
    For exmaple)
    ith sentence: xxx \n\n'''+con_x
    # print(cur_prompt)
    cur_try=0
    cur_text=None
    while cur_text is None and cur_try < try_limit:
        try:
            cur_try+=1
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": cur_prompt,
                    }
                ],
                model="llama3-70b-8192",
            )
            cur_text = chat_completion.choices[0].message.content
        except:
            sleep(sleep_time)
    with open(f'data/HC3_LLMs_L70b_para_mid.csv','a', newline='') as f:
        f.write(json.dumps({'idx':i,'human':text,'idx_list':idx_list,'generated':cur_text})+'\n')


with open('data/HC3_LLMs_L70b_para_mid.csv','r') as f:
    x = f.readlines()
    x = [json.loads(k) for k in x]

new_text_list=[]
for i in range(len(x)):
    text = x[i]['human']
    idx_list = x[i]['idx_list']
    para_texts = x[i]['generated'].split('sentence: ')[1:]
    tokened_sent = nltk.sent_tokenize(text)
    xfiltered = list(filter(lambda t: ' sentence: ' in t,x[i]['generated'].split('\n\n')[1:-1]))
    x_texts = [t.split('sentence: ')[1] for t in xfiltered]
    para_list = [int(re.search(r'\d+', k).group()) for k in xfiltered]
    try:
        for j,cur_id in enumerate(para_list):
            tokened_sent[cur_id] = x_texts[j]
    except:
        print(i)
        new_text_list.append(text)

    new_text_list.append(' '.join(tokened_sent))
for i in range(len(new_text_list)):
    pd_x.loc[i,'paraphrased'] = new_text_list[i]
pd_x.to_csv('data/HC3cleaned_para_LLMs.csv')