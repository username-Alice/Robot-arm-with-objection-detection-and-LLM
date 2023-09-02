import guidance
'''Contain the language model, prompt used in chat room'''
class LanguageModel():
    def __init__(self):
        super().__init__()
        llama = guidance.llms.LlamaCpp("source/model/gpt4-x-vicuna-13B.ggmlv3.q5_0.bin", tokenizer =  "ehartford/WizardLM-7B-Uncensored",n_gpu_layers=1)
        guidance.llm = llama
        self.model = guidance('''
        {{#block hidden=True}}
        {{#system~}}
        You are a helpful assistant. You do not ask questions.
        {{~/system}}
                                                
        {{#user~}}
        You will answer the user. At every step, I will provide you with the user input, as well as a object list informing you what object are available. You need to decide what object does the user need at the moment based on object list and user input. 
        There is no need to provide any extra information about the object. You can only use the object name that is listed in the object list and object name must be lowercase letters.
        Your answer should meet the following requirement:
        1. Always write the answer in one sentence.
        2. Always follow the format {'GPT': 'answer'}
        3. Never ask any type of questions.                     
        4. Always answer one and only one object and the object must be on the list.
        {{~/user}}
        {{/block}}                                     

        {{~! Then the conversation unrolls }}
        {{~#geneach 'conversation' stop=False}}
        {{#user~}}
        User: {{set 'this.input' (await 'input')}}
        Comment: Remember, the object you choose must be on the list. Object list is {{object_list}}. Do not response with questions.
        {{~/user}}

                        
        {{#assistant~}}
        {{gen 'this.response' temperature=0.2 max_tokens=300}}
        {{~/assistant}}
        {{~/geneach}}''')
    def answer_prompt(self,query,object_list):
        response = self.model(input=query, object_list = object_list)
        return response['conversation'][0]['response']

#Testing Testing


