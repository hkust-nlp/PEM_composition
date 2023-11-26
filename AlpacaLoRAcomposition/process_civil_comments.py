import pandas as pd

def process_cc():
    df=pd.read_csv("civil_comments.csv")
    breakpoint()
    df_new=df[df['split']=='test']
    df_new=df_new[df_new['toxicity']>0.8]
    df_new.to_csv("selected_test_comments.csv",index=False,header=True)

def compute_avetox():
    input_path="20230517_test_t_0.csv"
    df=pd.read_csv(input_path,header=None)
    df=df.T
    breakpoint()
    # for c in [2,3,5,6]:
    #     dfc=[float(i) for i in df[c]]
    #     print(sum(dfc)/len(df))
    #     if c in [2,5]:
    #         dfc_t=[it for it in dfc if it > 0.8]
    #         print(len(dfc_t))
    for c in df.columns:
        if c%2==1:
            dfc=[float(i) for i in df[c]]
            print(sum(dfc)/len(df))
            dfc_t=[it for it in dfc if it > 0.8]
            print(len(dfc_t))

# compute_avetox()

breakpoint()

path0="20230517_test_n_0.csv"
df0=pd.read_csv(path0,header=None)
df0=df0.iloc[[2,3]].reset_index(drop=True)

path1="20230517_n_1.csv"
df1=pd.read_csv(path1,header=None)
df1=df1.iloc[[1]].reset_index(drop=True)
df2=pd.concat([df0,df1]).reset_index(drop=True)

path3="nontox_score_v0.csv"
df3=pd.read_csv(path3,header=None)
df4=pd.concat([df3,df2]).reset_index(drop=True)

df4.to_csv("nontox_score_0517.csv",index=False,header=None)

def extend_size0():
    path0="20230517_test_t_0.csv"
    df0=pd.read_csv(path0,header=None)

    breakpoint()
    df0=df0.iloc[[0,10]].reset_index(drop=True)
    df2=pd.concat([df0,df100],axis=1)
    df2.columns = range(df2.columns.size)
    path3="nontox_score.csv"
    df3=pd.read_csv(path3,header=None)
    df4=pd.concat([df3,df2])
    df4=df4.reset_index(drop=True)
    df5 = df4.reindex(index=[0,1,4,2,3,5])
    path6="toxic_seed_v1_n.json"
    df6=pd.read_json(path6)
    text=list(df6['instruction'].values)
    df7=df5.T
    df7["text"]=text
    df5=df7.T
    df5 = df5.reindex(index=["text",0,1,4,2,3,5])
    df5.to_csv("nontox_score_0517.csv",index=False,header=None)

def extend_size():
    # path0="20230512_n_v0_0.csv"
    # path1="20230516_n_v0_0.csv"
    # df0=pd.read_csv(path0,header=None)
    # df1=pd.read_csv(path1,header=None)

    # breakpoint()

    # df3=df0.iloc[[0,1,20,21]]
    # df4=df3.reset_index(drop=True)
    # df5=pd.concat([df4,df1],axis=1)
    # df5.to_csv("nontox_score.csv",index=False,header=None)

    path0="20230515_n_1.csv"
    path1="20230516_n_1.csv"
    df0=pd.read_csv(path0,header=None)
    df1=pd.read_csv(path1,header=None)

    breakpoint()
    df0=df0.iloc[[0,10]].reset_index(drop=True)
    df2=pd.concat([df0,df1],axis=1)
    df2.columns = range(df2.columns.size)
    path3="nontox_score.csv"
    df3=pd.read_csv(path3,header=None)
    df4=pd.concat([df3,df2])
    df4=df4.reset_index(drop=True)
    df5 = df4.reindex(index=[0,1,4,2,3,5])
    path6="toxic_seed_v1_n.json"
    df6=pd.read_json(path6)
    text=list(df6['instruction'].values)
    df7=df5.T
    df7["text"]=text
    df5=df7.T
    df5 = df5.reindex(index=["text",0,1,4,2,3,5])
    df5.to_csv("nontox_score_v0.csv",index=False,header=None)

def find_incomplete_sentence(text):
    punctuations = ['.', '!', '?', '."','!"','?"', ".'","!'", "?'",".'",'...', '.)','!)','?)']  # 句子结束的标点符号列表

    # last_punctuation_index = max(text.rfind(p) for p in punctuations)  # 找到最后一个标点符号的索引
    last_punctuation_index, matched_punctuation = max((text.rfind(p), p) for p in punctuations)
    if last_punctuation_index != -1:
        incomplete_sentence = text[:last_punctuation_index + len(matched_punctuation)].strip()  # 切割未完成的部分并去除首尾空格
        return incomplete_sentence

    return text  # 若未找到句子结束的标点符号，则返回原字符串

def postprocess():
    input_path="20230517_test_t.csv"
    df=pd.read_csv(input_path,header=None)
    breakpoint()
    for i in range(len(df)):
        for j in range(len(df.columns)):
            text=df.iloc[i,j]
            text=str(text)
            text_edit=find_incomplete_sentence(text)
            df.iloc[i,j]=text_edit
            if text!=text_edit:
                print(text,'\n',text_edit)
                # breakpoint()
    df.to_csv("20230517_test_t_v0.csv",index=None,header=None)

# postprocess()