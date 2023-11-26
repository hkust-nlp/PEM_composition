import argparse
import os

import pandas as pd
from detoxify import Detoxify

import csv


def load_input_text(input_obj):
    """Checks input_obj is either the path to a txt file or a text string.
    If input_obj is a txt file it returns a list of strings."""

    if isinstance(input_obj, str) and os.path.isfile(input_obj):
        if input_obj.endswith(".txt"):
            text = open(input_obj).read().splitlines()
        elif input_obj.endswith(".csv"):
            text = pd.read_csv(input_obj,header=None)
            # breakpoint()
            text=text.T
            text=list(text[1].values)
        else:
            raise ValueError("Invalid file type: only txt & csv files supported.")
    elif isinstance(input_obj, str):
        text = input_obj
    else:
        raise ValueError("Invalid input type: input type must be a string or a txt/csv file.")
    return text


def run(model_name, input_obj, dest_file, from_ckpt, device="cpu"):
    """Loads model from checkpoint or from model name and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a txt file.
    """
    # breakpoint()
    if isinstance(input_obj, list): 
        text = input_obj
    else:
        text = load_input_text(input_obj)
    if model_name is not None:
        model = Detoxify(model_name, device=device)
    else:
        model = Detoxify(checkpoint=from_ckpt, device=device)
    res = model.predict(text)

    res_df = pd.DataFrame(res, index=[text] if isinstance(text, str) else text).round(5)
    print(res_df)
    print(res_df['toxicity'].mean())
    if dest_file is not None:
        res_df.index.name = "input_text"
        res_df.to_csv(dest_file)

    return res_df

def generate_toxic_instruction(data_path, save_path="datasets/"):
    df=pd.read_json(data_path)
    text=list(df['instruction'].values)
    res=run("original",text,None,None)
    for item in res.columns:
        df[item]=res[item].values
    breakpoint()
    df1=df[df['toxicity']>0.01]
    df2=df[df['toxicity']<=0.01]
    df.to_json(os.join(save_path,"toxic_seed_v01.json"),orient="records",indent=4)
    df1.to_json(os.join(save_path,"toxic_seed_v01_t.json"),orient="records",indent=4)
    df2.to_json(os.join(save_path,"toxic_seed_v01_n.json"),orient="records",indent=4)

def get_toxic_score(
    input_path="",
    save_path="",
):
    text = pd.read_csv(input_path,header=None)
    breakpoint()
    text=text.T
    for item in text.columns:
        # breakpoint()
        texts=list(text[item].values)
        texts=[str(text) for text in texts]
        res=run("original",texts,None,None)
        res_t=list(res['toxicity'].values)
        with open(save_path, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(texts)
            csvwriter.writerow(res_t)

def find_incomplete_sentence(text):
    punctuations = ['.', '!', '?', '."','!"','?"', ".'","!'", "?'",".'",'...', '.)','!)','?)']

    last_punctuation_index, matched_punctuation = max((text.rfind(p), p) for p in punctuations)
    if last_punctuation_index != -1:
        incomplete_sentence = text[:last_punctuation_index + len(matched_punctuation)].strip()
        return incomplete_sentence

    return text

def postprocess(
    input_path="",
    save_path="",
):
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
    df.to_csv(save_path,index=None,header=None)

# if __name__ == "__main__":
#     # args
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input",
#         type=str,
#         help="text, list of strings, or txt file",
#     )
#     parser.add_argument(
#         "--model_name",
#         default="unbiased",
#         type=str,
#         help="Name of the torch.hub model (default: unbiased)",
#     )
#     parser.add_argument(
#         "--device",
#         default="cpu",
#         type=str,
#         help="device to load the model on",
#     )
#     parser.add_argument(
#         "--from_ckpt_path",
#         default=None,
#         type=str,
#         help="Option to load from the checkpoint path (default: False)",
#     )
#     parser.add_argument(
#         "--save_to",
#         default=None,
#         type=str,
#         help="destination path to output model results to (default: None)",
#     )

#     args = parser.parse_args()

#     assert args.from_ckpt_path is not None or args.model_name is not None

#     if args.model_name is not None:
#         assert args.model_name in [
#             "original",
#             "unbiased",
#             "multilingual",
#         ]

#     if args.from_ckpt_path is not None and args.model_name is not None:
#         raise ValueError(
#             "Please specify only one model source, can either load model from checkpoint path or from model_name."
#         )
#     if args.from_ckpt_path is not None:
#         assert os.path.isfile(args.from_ckpt_path)
#     run(
#         args.model_name,
#         args.input,
#         args.save_to,
#         args.from_ckpt_path,
#         device=args.device,
#     )