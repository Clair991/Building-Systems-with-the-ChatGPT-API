# 1. LLM、ChatGTP API和Token
## 1.1 LLM是如何工作的
文本生成的过程：给定上文，模型生成下文。  
<img width="519" alt="1" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/b27f1376-9148-4bc8-98dc-defb34423ccd">

如何得到上述的LLM？主要还是使用监督学习。下面是一个餐馆评论情绪分类的训练和推理流程。  
<img width="519" alt="2" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/a67d6efd-1910-4d25-af44-b08db7389618">

LLM训练流程：样本数据X是句子的上文，样本标签Y是句子的下文。  
<img width="519" alt="3" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/f63da07f-4a13-4b9f-8f39-ebf8195ad709">

LLM类型：
- Base LLM（基础语言模型）
- Instruction Tuned-LLM（指令微调的大语言模型）

<img width="519" alt="4" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/8ae634b1-9285-4179-8242-2c7b8bb139de">

Base LLM 可以根据给出的上文，生成下文。但是对于问题则无法给出答案，上例中询问了Base LLM法国的首都是哪里这种诸如此类的问题，它没办法回答，而经过指令微调的大语言模型则可以完成这种问答任务，因为其在指令数据集上进行了微调，适配了问答这种下游任务。  
基础LLM的训练时间可能在几个月的时间，而经过指令微调的LLM根据指令数据集的规模在几天内就可以训练完。  
下面是从Base LLM到Instruction Tuned LLM的流程：  
<img width="519" alt="5" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/42d73483-e789-4878-a41b-8e684dab212d">

## 1.2 Tokens
如果让LLM去翻转一个单词，则会出错。  
~~~python
response = get_completion("Take the letters in lollipop \
and reverse them")
print(response)
~~~

<img width="715" alt="6" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/1d76d978-ddf2-453d-987a-e5818697b1e7">

为什么这么简单的任务，而功能强大的LLM却完成不了呢？实际上LLM训练过程中并不是预测的是严格意义上的字符，而是token。token的生成过程中会被划分成常见的词，这就可能导致一些生僻词容易被拆分。  
<img width="500" alt="7" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/f4beedf1-d8f2-40c5-81a4-4788e537c2a1">

在训练时，`lollipop`这个词实际上被分为了3个token：`l`，`oll`和`ipop`，所以一开始让模型将单词逆序就非常困难了。  
如果在单词的字母之间加上破折号，则可以逆序输出。  
~~~python
response = get_completion("""Take the letters in \
l-o-l-l-i-p-o-p and reverse them""")
print(response)
~~~
<img width="681" alt="8" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/011a0c2d-1bff-4f14-8c0b-15e0f654820a">

因为在训练时，这一串字符按照上述规则拆分为token了，其是最小粒度的，所以可以逆序输出。  
在英文文本的输入中，1 个token大概4个字符或者是3/4个单词。所以不同的语言模型会有不同数量的输入和输出token的数量限制。如果输入超过数量限制，则会抛出异常。gpt3.5-turbo模型的限制是4000个token。  
**输入通常被称作上下文（context），输出通常被称为补全（completion）。**

## 1.3 ChatGPT API
ChatGPT API 的调用方式：
~~~python
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
    )
    return response.choices[0].message["content"]
~~~
messages的结构：  
<img width="604" alt="9" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/4c2b7a53-bac2-4a52-a222-3bc6da7c8397">

ChatGPT API 中有三种不同的角色，其职责也不同。系统角色设定了LLM（助手）整体的语言风格，用户角色是使用者撰写的具体地指令，助手角色是LLM给出的响应。这样设计可以让无状态的API实现多轮对话中让模型能够利用历史会话信息当做上下文。  
token用量统计函数：
~~~python
def get_completion_and_token_count(messages, 
                                   model="gpt-3.5-turbo", 
                                   temperature=0, 
                                   max_tokens=500):
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    
    content = response.choices[0].message["content"]
    
    token_dict = {
'prompt_tokens':response['usage']['prompt_tokens'],
'completion_tokens':response['usage']['completion_tokens'],
'total_tokens':response['usage']['total_tokens'],
    }

    return content, token_dict
~~~
~~~python
messages = [
{'role':'system', 
 'content':"""You are an assistant who responds\
 in the style of Dr Seuss."""},    
{'role':'user',
 'content':"""write me a very short poem \ 
 about a happy carrot"""},  
] 
response, token_dict = get_completion_and_token_count(messages)
~~~

## 1.4 LLM构建应用的优势
LLM特别适用于非结构化数据，文本数据和视觉数据。与传统的监督学习建模的方式相比，其可以大大提升开发速度。  
<img width="518" alt="10" src="https://github.com/Clair991/Building-Systems-with-the-ChatGPT-API/assets/102845425/edc1c04f-8a5c-4c2e-b922-1ffa40ab7f8e">
