import google.generativeai as genai
import os
from config import GEMINI_API_KEY

api_key = GEMINI_API_KEY
genai.configure(api_key = api_key)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

response = chat.send_message("Please tell me a joke.")
print(response.text)