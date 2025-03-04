import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

class ShoppingChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small", dataset_path="../data/products_cleaned.json"):
        """
        Initialize the chatbot with the selected DialoGPT model.
        :param model_name:
        """
        print("Loading chatbot model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None #Store conversation history
        self.dataset_path = dataset_path

        #Load dataset dynamically
        self.categories, self.colors, self.brands = self.load_product_data()

        # User preferences storage
        self.user_preferences = {
            "category": None,
            "brand": None,
            "color": None,
            "price_range": None
        }

    def load_product_data(self):
        """
        Loads categories, colors and brands from dataset dynamically.
        :return:
        """
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                products = json.load(f)
        except FileNotFoundError:
            print(f"Error: {self.dataset_path} not found.")
            return set(), set(), set()

        categories = set()
        colors = set()
        brands = set()

        for product in products:
            if "category" in product and product["category"]:
                categories.add(product["category"].strip().lower())
            if "color" in product and product["color"]:
                colors.add(product["color"].strip().lower())  # Ensure colors are lowercase
            if "brand" in product and product["brand"]:
                brands.add(product["brand"].strip().lower())
    # Debugging: Print loaded colors
        print(f"Loaded Colors: {colors}")  # Check if "pink" is present

        return categories, colors, brands

    def extract_preferences(self, user_input):
        """
        Extracts user preferences (category, color, brand, price) dynamically from input.
        :param user_input:
        :return:
        """
        updated = False # Track if a preference is updated


        # Extract category
        for cat in self.categories:
            if cat.lower() in user_input.lower():
                self.user_preferences["category"] = cat
                updated = True

            # **Fuzzy match color**
        for color in self.colors:
            color_words = color.split()  # Split "Pink Nude" into ["Pink", "Nude"]
            for word in color_words:
                if word in user_input.lower():
                    self.user_preferences["color"] = color  # Store full color name
                    updated = True
                    break  # Stop once we find a match

        # Extract brand
        for brand in self.brands:
            if brand.lower() in user_input.lower():
                self.user_preferences["brand"] = brand
                updated = True

        #Extract price
        price_pattern =  r"\$(\d+)|(\d+\s?(dollars|euros|shekels))"
        match = re.search(price_pattern, user_input)
        if match:
            self.user_preferences["price_range"] = match.group(0)
            updated = True

        # Debugging: Print stored preferences
        print("Current Preferences:", self.user_preferences)

        return updated

    def generate_response(self, user_input):
        """
        Generates a chatbot response based on user input
        :param user_input:
        :return:
        """
        updated = self.extract_preferences(user_input) #Update stored preferences

        #If preferences are detected personalize response
        if updated:
            response = "Got it! You're looking for"
            if self.user_preferences["color"]:
                response += f" a {self.user_preferences['color']}"
            if self.user_preferences["category"]:
                response += f" {self.user_preferences['category']}"
            if self.user_preferences["brand"]:
                response += f" from {self.user_preferences['brand']}"
            if self.user_preferences["price_range"]:
                response += f" around {self.user_preferences['price_range']}"
            return response + ". Let me find the best options for you!"

         #Add a context to guide responses
        shopping_context = "You are a helpful AI shopping assistant. Your goal is to recommend beauty and makeup products based on user preferences."

        #Format input with context
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
                                           do_sample=True, temperature=0.7,
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
