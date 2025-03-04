import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ShoppingChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        """
        Initialize the chatbot with the selected DialoGPT model.
        :param model_name:
        """
        print("Loading chatbot model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None #Store conversation history

    def generate_response(self, user_input):
        """
        Generates a chatbot response based on user input
        :param user_input:
        :return:
        """

        #Add a context to guide responses
        shopping_context = "You are a helpful AI shopping assistant. Your job is to suggest beauty products based on the user's preferences."

        #Tokenize user input and append to chat history
        new_input = f"{shopping_context} User: {user_input} AI:"
        new_input_ids = self.tokenizer.encode(new_input + self.tokenizer.eos_token, return_tensors="pt")

        # Concatenate with previous chat history (if available)
        if self.chat_history_ids is None:
            self.chat_history_ids = new_input_ids
        else:
            self.chat_history_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)

        #Generate response with attention mask
        attention_mask = torch.ones(self.chat_history_ids.shape, dtype=torch.long)
        response_ids = self.model.generate(self.chat_history_ids, max_length=200,
                                           pad_token_id=self.tokenizer.eos_token_id, attention_mask=attention_mask,
                                           temperature=0.7,
                                           # Controls creativity (lower = more predictable, higher = more random)
                                           top_p=0.9  # Nucleus sampling (filters unlikely words)
                                           )

        #Decode response and update chat history
        response = self.tokenizer.decode(response_ids[:, self.chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    chatbot= ShoppingChatbot()
    print("Chatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        response = chatbot.generate_response(user_input)
        print(f"Chatbot: {response}")
