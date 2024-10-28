import os
import json
import boto3
from pdf2image import convert_from_bytes
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableAssign
import io
import base64
from operator import itemgetter
import streamlit as st
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import BedrockChat
from langchain_core.runnables import chain
import pandas as pd

aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Reading prompt
with open("prompt.txt", "r") as f:
    prompt = f.read()


def convert_pdf_to_images(pdf_paths: list):
    inputs = list()
    for each_pdf in pdf_paths:
        pdf_name = each_pdf.name
        images = convert_from_bytes(each_pdf.getvalue(), dpi=300)
        for idx, image in enumerate(images):
            image_dict = dict()
            image_file = io.BytesIO()
            image.save(image_file, format="JPEG")
            image_file.seek(0)
            image_dict["pdf_name"] = pdf_name
            image_dict["page_number"] = f"page_{idx + 1}"
            image_dict["image_path"] = image_file
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
                {"type": "text", "text": "<image>"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64_string,
                    },
                },
                {"type": "text", "text": "</image>"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages


def get_body(messages):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_k": 250,
            "top_p": 0.999,
            "stop_sequences": ["\n\nHuman"],
            "messages": messages,
        }
    )
    return body


def get_response(body):
    try:
        modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
        contentType = "application/json"
        accept = "application/json"

        response = bedrock_client.invoke_model(
            modelId=modelId, contentType=contentType, accept=accept, body=body
        )
        response_body = json.loads(response.get("body").read())
        return response_body
    except Exception as e:
        st.error(f"Error during response parsing: {e}")
        return None


@chain
def extract_text_chain(inputs):
    image_b64 = image_to_base64(inputs["image_path"])
    messages = get_messages(image_b64)
    body = get_body(messages)
    response_body = get_response(body)

    if (
        response_body
        and "content" in response_body
        and isinstance(response_body["content"], list)
    ):
        inputs["text"] = response_body["content"][0].get("text", "")
        inputs["input_tokens"] = response_body.get("usage", {}).get("input_tokens", 0)
        inputs["output_tokens"] = response_body.get("usage", {}).get("output_tokens", 0)
    else:
        inputs["text"] = ""
        inputs["input_tokens"] = 0
        inputs["output_tokens"] = 0

    return inputs


mapper = RunnableParallel({"output": itemgetter("text") | JsonOutputParser()})
output_parser_chain = RunnableAssign(mapper)

final_chain = extract_text_chain | output_parser_chain


def get_aggregated_dataframe(pdf_files):
    inputs = convert_pdf_to_images(pdf_paths=pdf_files)
    response = final_chain.batch(inputs)

    columns = [
        "pdf_name",
        "page_number",
        "investment_name",
        "ticker",
        "type",
        "Present_unit_price",
        "date",
    ]
    df_main = pd.DataFrame(columns=columns)

    for each in response:
        if "output" in each:
            df = pd.DataFrame(each["output"])
            df["pdf_name"] = each.get("pdf_name", "")
            df["page_number"] = each.get("page_number", "")
            df_main = pd.concat([df_main, df], axis=0)

    df_main = df_main.reset_index(drop=True)
    return df_main
