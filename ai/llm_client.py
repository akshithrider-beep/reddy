from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from prompts.system_prompts import TUTOR_SYSTEM_PROMPT, GENERAL_TUTOR_PROMPT

load_dotenv()

# Initialize GenAI client
# Requires GEMINI_API_KEY in .env
client = genai.Client()

def ask_tutor(query: str, question_context: dict = None) -> str:
    try:
        if question_context:
            # Format context string
            context_str = f"Question: {question_context.get('question')}\n"
            context_str += f"Options: {question_context.get('options')}\n"
            context_str += f"Correct Option Index: {question_context.get('correct_option_index')}\n"
            context_str += f"Explanation from book: {question_context.get('explanation')}\n"
            
            sys_instruct = TUTOR_SYSTEM_PROMPT.format(context=context_str)
        else:
            sys_instruct = GENERAL_TUTOR_PROMPT

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.3, # Low temperature for factual engineering answers
            )
        )
        return response.text
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return f"Error communicating with AI Tutor: {str(e)}"
