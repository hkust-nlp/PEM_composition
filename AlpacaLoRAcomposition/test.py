import os
import openai
from generate_instructions_openai import encode_prompt, generate_instruction_following_data, clear_from_json,sum_from_json,clear_http, get_help_score

openai.organization = "xxx"
openai.api_key = "xxx"

generate_instruction_following_data(source_output_path="./selected_test_comments.csv",regen_path="seed2.json",start_num=210,num_instructions_to_generate=200, model_name="gpt-4")#text-davinci-003
generate_instruction_following_data(regen_path="regen_v3.json",num_prompt_instructions=5, start_num=30000)# ,num_instructions_to_generate=20000)#, model_name="gpt-4")#text-davinci-003
# print(encode_prompt([], "You support Trump...an avowed racist. Why?"))

clear_from_json("regen_v01.json","regen_v01.json")
paths=["regen_v01.json","regen_v12.json","regen_v21.json","regen_v31.json"]
sum_from_json(["regen_all_v23.json"],"regen_all_v2.json")
clear_http("regen_all_v0.json","regen_all_v21.json")

get_help_score("toxic_seed_v01_n.json","20230516_n_v0.csv","20230516_n_1.csv")
get_help_score("toxic_seed_v1_n.json","20230517_test_n_v0.csv","20230517_n_1.csv")