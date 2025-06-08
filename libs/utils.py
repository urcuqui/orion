import re

def clean_response_deepseek(text):    
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()