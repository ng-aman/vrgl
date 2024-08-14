
import json
import boto3
from pdf2image import convert_from_bytes
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableAssign
import io
import base64
from operator import itemgetter
# from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import BedrockChat

from langchain_core.runnables import chain

import pandas as pd

bedrock_client = boto3.client(
  service_name='bedrock-runtime',
  region_name='us-east-1'
)

# reading prompt
with open("prompt.txt","r") as f:
    prompt= f.read()

def convert_pdf_to_images(pdf_paths: list):
    
    # declaring all pdf paths list
    inputs= list()
    # Convert PDF to images
    for each_pdf in pdf_paths:
        # getting pdf name
        pdf_name= each_pdf.name
        # converting to images
        images = convert_from_bytes(each_pdf.getvalue(), dpi=300)
        # Save each page as an in-memory image
        # image_files= list()
        for idx,image in enumerate(images):
            image_dict= dict()
            image_file = io.BytesIO()
            image.save(image_file, format='JPEG')
            image_file.seek(0)
            # image_files.append(image_file)
            image_dict['pdf_name']= pdf_name
            image_dict['page_number']= f'page_{idx+1}'
            image_dict['image_path']= image_file

            #appending images to pdf paths
            inputs.append(image_dict)
    
    return inputs

def image_to_base64(image_file):
    image_b64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return image_b64_string

def get_messages(image_b64_string):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<image>"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64_string
                    }
                },
                {
                    "type": "text",
                    "text": "</image>"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    return messages

def get_body(messages):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "temperature": 0.1,
        "top_k": 250,
        "top_p": 0.999,
        "stop_sequences": ["\n\nHuman"],
        "messages": messages
    })
    return body

def get_response(body):
    print("getting a call")
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
    # modelId = 'anthropic.claude-3-haiku-20240307-v1:0'
 
    contentType = 'application/json'
    accept = 'application/json'
 
    response = bedrock_client.invoke_model(
        modelId=modelId,
        contentType=contentType,
        accept=accept,
        body=body
    )
    response_body = json.loads(response.get('body').read())
    print(response_body)
    return response_body

@chain
def extract_text_chain(inputs):
    image_b64 = image_to_base64(inputs['image_path'])
    messages = get_messages(image_b64)
    body = get_body(messages)
    response_body = get_response(body)
    inputs['text']= response_body['content'][0]['text']
    inputs['input_tokens']= response_body['usage']['input_tokens']
    inputs['output_tokens']= response_body['usage']['output_tokens']
    return inputs

# creating output parser chain
mapper= RunnableParallel({"output":itemgetter("text") | JsonOutputParser()})
output_parser_chain = RunnableAssign(mapper)

# creating final chain
final_chain = extract_text_chain | output_parser_chain

def get_aggregated_dataframe(pdf_files):
    # getting inputs
    inputs= convert_pdf_to_images(pdf_paths=pdf_files)
    # getting response
    response= final_chain.batch(inputs)
    # converting to excel
    columns= ['pdf_name','page_number','investment_name', 'ticker', 'type', 'Present_unit_price', 'date']
    df_main= pd.DataFrame(columns= columns)
    for each in response:
        # creating pandas dataframe
        df= pd.DataFrame(each['output'])
        print(df.columns)
        print(df.shape)
        print("="*100)
        df['pdf_name']= each['pdf_name']
        df['page_number']= each['page_number']
        # df= df[columns]

        # appending temp df to main df
        df_main = pd.concat([df_main,df],axis=0)
    
    # reset index
    df_main= df_main.reset_index(drop=True)
    
    return df_main
