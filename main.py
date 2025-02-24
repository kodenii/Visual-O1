from gpt4v import Agent
from PIL import Image
from time import sleep
import json
import random
from file_utils import read_json, read_jsonl
import argparse
from inference import run

gpt_config = read_json("gpt_config.json")
MODEL_NAME = gpt_config["MODEL_NAME"]
API_KEY = gpt_config["API_KEY"]
API_VERSION = gpt_config["API_VERSION"]
AZURE_ENDPOINT = gpt_config["AZURE_ENDPOINT"]


def get_instruction(args):
    prompts = read_json("instructions/prompt.json")
    data_name = args.data_name
    dataset_path = f"data/{data_name}"
    OPTIMIZER_INSTRUCTION = prompts["OPTIMIZER_INSTRUCTION"]
    OPTIMIZER_INIT = prompts["OPTIMIZER_INIT"]
    OPTIMIZER_UPDATE = prompts["OPTIMIZER_UPDATE"]

    opt_agent = Agent(MODEL_NAME, API_KEY, API_VERSION, AZURE_ENDPOINT, OPTIMIZER_INSTRUCTION)

    new_instruction = opt_agent.chat(text=OPTIMIZER_INIT)
    img_base_dir = "./images/"
    dataset = read_jsonl(dataset_path)
    with open("instructions/instructions.jsonl", 'a') as f:
        json.dump({"instruction": new_instruction}, f)
        f.write("\n")
    for epoch in range(4):
        disambiguate_agent = Agent(MODEL_NAME, API_KEY, API_VERSION, AZURE_ENDPOINT, new_instruction)
        data = dataset[random.randint(0, len(dataset)-1)]
        user_input = data['sent']
        img_name = data['img_name']
        img_dir = img_base_dir + img_name
        image = Image.open(img_dir)
        result = disambiguate_agent.chat(text=user_input, image=image)
        print("Instruction:", new_instruction)
        print("original text:", user_input)
        print("disambiguous result:", result)
        print("=======================")

        new_instruction = opt_agent.chat(text=OPTIMIZER_UPDATE.format(user_input, result), image=image)
        with open("instructions/instructions.jsonl", 'a') as f:
            json.dump({"instruction": new_instruction}, f)
            f.write("\n")
            
    return new_instruction

def disambiguate(instruction, args):
    data_name = args.data_name
    dataset_path = f"data/{data_name}"
    dataset = read_jsonl(dataset_path)
    for data in dataset:
        agent = Agent(MODEL_NAME, API_KEY, API_VERSION, AZURE_ENDPOINT, instruction)
        sent = data["sent"]
        user_input = sent
        img_name = data["img_name"]
        img_base_dir = "./images/"
        image_path = img_base_dir + img_name
        image = Image.open(image_path)
        result = agent.chat(text=user_input, image=image)

        data["disambiguous_sent"] = result
        with open(f"data/deblur_{data_name}", "a") as f:
            json.dump(data, f)
            f.write("\n")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="data.jsonl")
    parser.add_argument("--mode", type=str, default="empirical")
    args = parser.parse_args()
    assert args.mode in ["empirical", "instantial"]
    mode = args.mode
    if mode == "empirical":
        instruction = get_instruction(args)
        disambiguate(instruction, args)
    run(args.data_name, args.mode)