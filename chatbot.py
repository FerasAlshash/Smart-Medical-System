from flask import request, jsonify
import re
from langchain.prompts.chat import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from chat_db import Conversation, Message, db
import uuid
from datetime import datetime
import speech_recognition as sr

# Initialize the chatbot model and memory
model = ChatMistralAI(
    mistral_api_key="22MIvQ0KgV5u0ZvxZdecv5wmwAwWhMPZ",
    model_name="pixtral-12b-2409"
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def format_response(response_content):
    # Replace **word** with bold word
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'\1', response_content)
    return formatted_response

def save_to_db(conversation_id, content, is_user):
    with db:
        # Get or create conversation
        conversation, created = Conversation.get_or_create(
            conversation_id=conversation_id,
            defaults={'start_time': datetime.now()}
        )

        # Update last_update time
        conversation.last_update = datetime.now()
        conversation.save()

        # Create message
        Message.create(
            conversation=conversation,
            content=content,
            is_user=is_user
        )

def chatbot():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id')

        # Generate new conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Save user's question to database
        save_to_db(conversation_id, question, is_user=True)

        # Add user's question to conversation memory
        memory.chat_memory.add_user_message(question)

        # Create Prompt using conversation history
        prompt = ChatPromptTemplate.from_template("""
You are now a medical assistant AI. Your task is to provide accurate, reliable, and compassionate responses to users' health-related questions . Always ensure your answers are clear, concise, and free of jargon or irrelevant words.
### Key Guidelines:
1. **Language:** Use proper grammar, spelling, and avoid any unclear terms or irrelevant words.
2. **Tone:** Maintain a polite, supportive, and empathetic tone at all times.
3. **Accuracy:** Base your responses on credible medical knowledge, avoiding any confusing or irrelevant terms.
4. **Clarity:** Ensure your answers are easy to understand, using bullet points for better readability.
5. **Context Awareness:** Use the chat history provided to understand the context of the current question. Do not repeat previous answers unless necessary. Instead, focus on providing new insights based on the user's follow-up questions.
6. **Avoid Repetition:** Do not use the same words or phrases repeatedly. Provide diverse and meaningful responses.
7. **Provide Practical Advice:** When answering questions about blood pressure or other health issues, always provide practical advice such as lifestyle changes, diet tips, or consulting a doctor.
### Examples:
#### Example 1:
User: What are the values ​​considered high for blood pressure?
AI: Hello! Your blood pressure is considered high if:
- Systolic (upper) blood pressure is higher than 140 mmHg.
- Diastolic (lower) blood pressure is higher than 90 mmHg.
Here are some tips to lower your blood pressure:
- Exercise regularly.
- Reduce your salt intake.
- Maintain a healthy weight.
- Avoid stress and anxiety.
#### Example 2:
User: What should I do if my blood pressure is high?
AI: Based on your previous question about high blood pressure, here are some tips:
- Make sure to monitor your blood pressure readings regularly.
- Reduce your intake of salty foods.
- Try to relax and practice deep breathing techniques.
- Consult your doctor for appropriate treatment.
### Response Structure:
1. Start with a warm greeting.
2. Address the user's question clearly and concisely.
3. Provide relevant information using bullet points for better readability.
4. End with a friendly note encouraging further consultation if needed.
### Chat History:
{chat_history}
### Current Question:
Question: {input}
""")
        formatted_prompt = prompt.format_messages(
            input=question,
            chat_history="\n".join([msg.content for msg in memory.chat_memory.messages])
        )
        response = model.invoke(formatted_prompt)
        response_content = response.content.strip()

        # Format the response for HTML display
        formatted_response = format_response(response_content)

        # Save AI's response to database
        save_to_db(conversation_id, formatted_response, is_user=False)

        # Add AI's response to conversation memory
        memory.chat_memory.add_ai_message(formatted_response)

        return jsonify({'response': formatted_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def recognize_speech(audio_data, language='en-US'):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        return "Sorry, I did not understand that."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
