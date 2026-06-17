import speech_recognition as sr
import asyncio
import edge_tts
from playsound import playsound
import os

from ollama import chat
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


async def speak(text):
    file = "voice.mp3"

    tts = edge_tts.Communicate(
        text=text,
        voice="en-US-AriaNeural"
    )

    await tts.save(file)

    playsound(file)

    if os.path.exists(file):
        os.remove(file)


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

recognizer = sr.Recognizer()

print("AI Voice Meeting Agent Started...")


while True:
    try:

        choice = input(
            "\nPress ENTER to ask a question or type 'exit' to quit: "
        )

        if choice.lower() == "exit":
            print("Goodbye!")
            break

        with sr.Microphone() as source:

            print("Recording... Speak now")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=0.3
            )

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=8
            )

        question = recognizer.recognize_google(audio)

        print(f"\nYou: {question}")

        # Voice Exit Commands
        if question.lower() in [
            "exit",
            "quit",
            "stop",
            "goodbye",
            "bye"
        ]:
            print("Goodbye!")

            asyncio.run(
                speak("Goodbye, have a nice day.")
            )

            break

        docs = db.similarity_search(
            question,
            k=3
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
Answer the user's question directly.

Question:
{question}

Meeting Context:
{context}

Important:
- Use meeting context ONLY if it is relevant to the question.
- If the question is unrelated to the meeting context, completely ignore the meeting context.
- Do not mention whether information was found or not found in the meeting context.
- Do not say phrases like:
  'There is no information in the meeting context'
  'Based on the meeting context'
  'According to my general knowledge'
- Just give the answer naturally.
"""

        print("\nSearching meeting notes...")

        response = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        answer = answer[:500]

        print(f"\nAgent: {answer}")

        print("\nSpeaking...")

        asyncio.run(
            speak(answer)
        )

        print("Speech Completed")

    except sr.WaitTimeoutError:
        print("No speech detected.")

    except sr.UnknownValueError:
        print("Could not understand audio.")

    except Exception as e:
        print(f"Error: {e}")