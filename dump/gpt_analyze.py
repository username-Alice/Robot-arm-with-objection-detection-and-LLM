from gpt4all import *
import os
#prompt analysis
def create_combined_prompt(prompt, object_list):
    combined_prompt ="Must choose an object from the object list and best matches with the statement,be concise, give object name that has exact same format as shown in the object list."
    combined_prompt += "Given the statement: "+ prompt+". " 
    combined_prompt += "And the object list: ["
    for cls in object_list:
        if (cls != object_list[0]):
            combined_prompt += ","
        combined_prompt += cls
    combined_prompt += "]"
    print(combined_prompt)
    return combined_prompt

def gpt(prompt, object_list):
    #nous-hermes-13b.ggmlv3.q4_0.bin
    #ggml-vicuna-13b-1.1-q4_2.bin
    #ggml-mpt-7b-instruct.bin
    combined_prompt = create_combined_prompt(prompt, object_list)
    Class = ""
    if object_list != None:
        absolute_path = os.path.dirname(__file__)
        model_path = os.path.join(absolute_path, "model/ggml-vicuna-13b-1.1-q4_2.bin")
        model = GPT4All(model_path)
        #output = model.generate(combined_prompt, max_tokens=1000)
        """with model.chat_session():
            tokens = list(model.generate(prompt=combined_prompt, top_k=1, streaming=False, temp=0.4))
            answer = ''.join(tokens)
            print(answer)
            for cls in object_list:
                if (cls in answer):
                    print(cls)"""
        tokens = model.generate(prompt = combined_prompt,top_k = 1,streaming=False, temp = 0.2)
        answer = ''.join(tokens)
        print(answer)
        for cls in object_list:
            if (cls in answer):
                print("cls = ", cls)
                Class = cls
    else:
        answer = "No object"
    return answer, Class