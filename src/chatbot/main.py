from textblob import TextBlob

# define intents and their corresponding responses based on keywords
intents = {
    "hours": {
        "keywords": ["hours", "open", "close"],
        "response": "We are open from 9 AM to 5 PM, Monday to Friday."
    },
    "return": {
        "keywords": ["return", "refund", "money back"],
        "response": "I'd be happy to help you with the return process. Let me transfer you to a live agent."
    }
}

def get_response(message):
    message = message.lower()

    for _, intent_data in intents.items():
        if any(keyword in message for keyword in intent_data["keywords"]):
            return intent_data["response"]

    senmtiment = TextBlob(message).sentiment.polarity

    return ("That's great to hear!" if senmtiment > 0 else
            "I'm sorry to hear that. How can I help?" if senmtiment < 0 else
            "I see. Can you tell me more about that?")


def chat():
    print("Chatbot: Hi. How can I help you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nChatbot: Thank you for chatting with us. Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()