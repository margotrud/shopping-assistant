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
        print("Loading chatbot model... This may take a few seconds...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None #Store conversation history
        self.dataset_path = dataset_path

        #Load dataset dynamically
        self.categories, self.colors, self.brands, self.color_groups = self.load_product_data()

        # User preferences storage
        self.user_preferences = {
            "category": None,
            "brand": None,
            "color": None,
            "color_group": None, #General color
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

        categories, colors, brands = set(), set(), set()
        color_groups = {}

        # Predefined core colors
        core_colors = {"pink", "red", "beige", "purple", "peach", "brown", "blue"}

        for product in products:
            if "category" in product and product["category"]:
                categories.add(product["category"].strip().lower())
            if "color" in product and product["color"]:
                color_name = product["color"].strip().lower()
                colors.add(color_name)

                # Find which core color (if any) is in this name
                assigned = False
                for core_color in core_colors:
                    if core_color in color_name:
                        if core_color not in color_groups:
                            color_groups[core_color] = set()
                        color_groups[core_color].add(color_name)
                        assigned = True

                #If no core colour was found, store as is:
                if not assigned:
                    if "other" not in color_groups:
                        color_groups["other"] = set()
                    color_groups["other"].add(color_name)


            if "brand" in product and product["brand"]:
                brands.add(product["brand"].strip().lower())

        print(f"Loaded {len(categories)} categories, {len(colors)} colors, {len(brands)} brands.")  # Debugging
        print(f"Color Groups: {color_groups}")  # Debugging

        return categories, colors, brands, color_groups


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

            # Check if the user specified a general color (e.g. "pink")
        for base_color, shades in self.color_groups.items():
            if base_color in user_input.lower():
                self.user_preferences["color_group"] = base_color # Offer all shades of this base color
                self.user_preferences["color"] = None # Clear specific color
                updated = True
                break

        # Check if the user specified a specific color variation (e.g., "pink nude")
        for color in self.colors:
            if color in user_input.lower():
                self.user_preferences["color"] = color # Store specific shade
                self.user_preferences["color_group"] = None # Remove general color
                updated = True
                break

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
            # If a category is specified, mention it, otherwise keep it neutral
            category_text = f"{self.user_preferences['category']}" if self.user_preferences["category"] else "products"

            if self.user_preferences["color_group"]:
                shades = ", ".join(self.color_groups[self.user_preferences["color_group"]])
                response += f"{category_text} in shades of {self.user_preferences['color_group']} ({shades})"
            elif self.user_preferences["color"]:
                response += f" a {self.user_preferences['color']}"

            if self.user_preferences["brand"]:
                response += f" from {self.user_preferences['brand']}"
            if self.user_preferences["price_range"]:
                response += f" around {self.user_preferences['price_range']}"
            return response + ". Let me find the best options for you!"

         #Add a context to guide responses
        shopping_context = "You are a helpful AI shopping assistant. Your goal is to recommend beauty and makeup products based on user preferences."

        #Format input with context
        new_input = f"{shopping_context}\nUser: {user_input}\nAI:"
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
