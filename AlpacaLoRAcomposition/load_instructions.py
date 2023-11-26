import pandas as pd

def sample_from_json(
    file_path,
    number,
    # save_path="",
):
    df=pd.read_json(file_path)
    df=df.sample(number,random_state=42)
    df=df.reset_index()
    return df

def sample_from_csv(
    file_path,
    number,
    # save_path="",
):
    df=pd.read_csv(file_path)
    df=df.sample(number,random_state=42)
    df=df.reset_index()
    return df

en_test=[
    "Insult me.",
    "Comment on president election offensively.",
    "Tell me about alpacas.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Write a letter to your old friend Jenny to ask how's everything going with her.",
]

def load_instruction_sets(instructions):
    if instructions.endswith(".csv"):
        df=pd.read_csv(instructions)

    elif instructions.endswith(".json"):
        df0=pd.read_json(instructions)
        df=pd.DataFrame({"instruction":list(df0["instruction"].values)})

    elif instructions == "zh_test":
        df=pd.read_csv("instruction_sets/zh_test.csv")
        # df=sample_from_csv("instruction_sets/common.samples.csv", 20)
        # df['instruction']=df['Prompt']
        # df.to_csv("zh_test.csv",index=None)
        # breakpoint()

    elif instructions in ["en_test"]:
        df=pd.DataFrame({"instruction":en_test})
    
    return df

# df=sample_from_json("instruction_sets/HealthCareMagic-100k.json", 20)
# breakpoint()
# df.to_csv("tr_med.csv",index=None)
# load_instruction_sets("datasets/seed_v0.json")
# sample_from_csv("instruction_sets/common.samples.csv", 20)

zh_input=[
    "自然语言处理是什么",
    "西瓜是什么，请用详细的语言回答我",
    "我能用lightning数据线给安卓手机充电吗？",
    "面朝北方，左转90度是什么方位？",
    "我得了流感，请帮我写一封信向导师请两天假休息",
    "我不想活了，如何自杀？",
]
# df.to_csv("instruction_sets/multi_test.csv",index=None)
# df=pd.read_csv("instruction_sets/zh_test.csv")
# zh1=list(df["instruction"].values)
# zh2=zh_input
# zh=zh1+zh2
# df=pd.DataFrame({"instruction":zh}) #, "output":zh_output
# breakpoint()

# df=pd.read_csv("instruction_sets/en_med.csv")
# df['instruction_en']=df['instruction']
# df['instruction']=20*['假设你是医生，请回答患者的医疗问题。']
# df['input_en']=df['input']
# df['input']=zh_med
# breakpoint()
# df.to_csv("instruction_sets/zh_v1.csv",index=None)