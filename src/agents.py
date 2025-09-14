# In src/agents.py

import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from src.graph import AgentState
from dotenv import load_dotenv
load_dotenv()


os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)

# Initialize models
# For a fully open-source implementation, these can be replaced with models like Llama-3 and a vision model like Llava.
# This example uses OpenAI for simplicity and performance.
text_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
vision_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3) # gpt-4o is a powerful LVLM

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- Agent Node Implementations ---

def general_agent_node(state: AgentState):
    print("---EXECUTING GENERAL AGENT---")
    question = state['question']
    retrieved_texts = state['retrieved_texts']
    retrieved_images = state['retrieved_images']

    text_context = "\n\n".join([doc.page_content for doc in retrieved_texts])
    
    image_parts = []
    for img_path in retrieved_images:
        base64_image = encode_image(img_path)
        image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

    prompt = f"""
    You are an advanced agent capable of analyzing both text and images.
    Your task is to use both the textual and visual information provided to answer the user's question accurately.
    
    User Question: {question}
    
    Textual Context:
    ---
    {text_context}
    ---
    
    Analyze the provided images and the text to formulate a preliminary answer.
    """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt}
    ] + image_parts)
    
    response = vision_model.invoke([message])
    return {"general_agent_response": response.content}

def critical_agent_node(state: AgentState):
    print("---EXECUTING CRITICAL AGENT---")
    question = state['question']
    retrieved_texts = state['retrieved_texts']
    retrieved_images = state['retrieved_images']
    general_response = state['general_agent_response']

    text_context = "\n\n".join([doc.page_content for doc in retrieved_texts])

    class CriticalInfo(BaseModel):
        critical_text: str = Field(description="The most critical sentences or phrases from the text context that are essential to answer the question.")
        critical_image_description: str = Field(description="A detailed textual description of the most critical visual elements or regions in the images that are essential to answer the question.")

    structured_llm = text_model.with_structured_output(CriticalInfo)
    
    prompt = f"""
    You are a critical analysis agent. Your role is to identify the most crucial pieces of information from the provided context to answer the user's question.
    Analyze the user's question, the retrieved text, and the preliminary answer to extract the most salient points.
    
    User Question: {question}
    Preliminary Answer: {general_response}
    
    Retrieved Text Context:
    ---
    {text_context}
    ---
    
    Based on all the above, extract the critical textual information and provide a description of the critical visual information needed from the images to formulate a final, accurate answer.
    """
    
    response = structured_llm.invoke(prompt)
    
    return {"critical_text_info": response.critical_text, "critical_image_info": response.critical_image_description}

def text_agent_node(state: AgentState):
    print("---EXECUTING TEXT AGENT---")
    question = state['question']
    retrieved_texts = state['retrieved_texts']
    critical_text = state['critical_text_info']

    text_context = "\n\n".join([doc.page_content for doc in retrieved_texts])
    
    prompt = f"""
    You are a specialized text analysis agent. Your job is to perform a deep analysis of the provided text to answer the user's question.
    Focus your analysis on the critical information identified.
    
    User Question: {question}
    
    Critical Textual Information to Focus On:
    ---
    {critical_text}
    ---
    
    Full Text Context:
    ---
    {text_context}
    ---
    
    Provide a detailed, text-based answer based on your focused analysis.
    """
    
    response = text_model.invoke(prompt)
    return {"text_agent_response": response.content}

def image_agent_node(state: AgentState):
    print("---EXECUTING IMAGE AGENT---")
    question = state['question']
    retrieved_images = state['retrieved_images']
    critical_image_info = state['critical_image_info']

    image_parts = []
    for img_path in retrieved_images:
        base64_image = encode_image(img_path)
        image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

    prompt = f"""
    You are an advanced image processing agent specialized in analyzing and extracting information from images.
    Your task is to answer the user's question based on a focused analysis of the provided images, guided by the critical visual information.
    
    User Question: {question}
    
    Critical Visual Information to Focus On:
    ---
    {critical_image_info}
    ---
    
    Analyze the images, paying close attention to the regions or features highlighted by the critical information, and provide a visually-grounded answer.
    """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt}
    ] + image_parts)
    
    response = vision_model.invoke([message])
    return {"image_agent_response": response.content}

def summarizing_agent_node(state: AgentState):
    print("---EXECUTING SUMMARIZING AGENT---")
    question = state['question']
    general_response = state['general_agent_response']
    text_response = state['text_agent_response']
    image_response = state['image_agent_response']

    prompt = f"""
    You are a summarizing agent. Your task is to synthesize the findings from multiple specialized agents into a single, final, and comprehensive answer.
    Analyze the individual agent answers, identify commonalities, resolve discrepancies, and construct a final answer that leverages the collective intelligence of the system.
    
    User Question: {question}
    
    Preliminary Answer (from General Agent):
    ---
    {general_response}
    ---
    
    Text-Based Answer (from Text Agent):
    ---
    {text_response}
    ---
    
    Image-Based Answer (from Image Agent):
    ---
    {image_response}
    ---
    
    Based on all the provided answers, synthesize the final, most accurate, and complete response.
    """

    response = text_model.invoke(prompt)
    return {"final_answer": response.content}