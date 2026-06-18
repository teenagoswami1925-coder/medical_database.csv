import os
from google import genai

# The client automatically picks up the GEMINI_API_KEY environment variable
client = genai.Client(api_key= "AQ.Ab8RN6J7FC9hEKd2IYRzfs48I8bZfrXtNBsvWIBTZ0FDvwaVkg")

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Confirming connection. Respond with the word "Success!" if you can read this.',
)

print(response.text)