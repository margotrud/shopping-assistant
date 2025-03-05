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
        self.products, self.categories, self.colors, self.brands, self.color_groups = self.load_product_data()

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

        return products, categories, colors, brands, color_groups

    def find_matching_products(self):
        """
        Finds products that match user preferences (category, color, brand).
        :return:
        """
        matching_products = []

        for product in self.products:
            # Exclude brands, categories, colors and price
            if "exclude" in self.user_preferences:
                exclude_data = self.user_preferences["exclude"]

                if product["brand"].lower() in exclude_data["brands"]:
                    continue #Skip excluded brand
                if product["category"].lower() in exclude_data["categories"]:
                    continue #Skip excluded category
                if product["color"].lower() in exclude_data["colors"]:
                    continue #Skip excluded color

                if exclude_data["price"] and float(product.get("price", 0)) < exclude_data["price"]:
                    continue # Skip if price is below excluded threshold


            # if only brand is specified, return all products from that brand
            if self.user_preferences["category"] and product["category"].lower() != self.user_preferences["category"]:
                continue

            if self.user_preferences["color"] and product["color"].lower() != self.user_preferences["color"]:
                continue

            if self.user_preferences["color_group"]:
                if product["color"].lower() not in self.color_groups.get(self.user_preferences["color_group"], {}):
                    continue

            if self.user_preferences["brand"] and product["brand"].lower() != self.user_preferences["brand"]:
                continue

            if self.user_preferences["price_range"]:
                product_price = float(product.get("price", 0))
                max_price = float(self.user_preferences["price_range"].replace("$", ""))
                if product_price > max_price:
                    continue
            # Add matching product
            matching_products.append(product)

        return matching_products

    def extract_preferences(self, user_input):
        """
        Extracts user preferences (category, color, brand, price) dynamically from input.
        :param user_input:
        :return:
        """
        updated = False # Track if a preference is updated

        # Track what type of preference was detected
        detected_preferences = {"brands": False, "categories": False, "colors": False, "color_group": False, "price_range": False}

        # Ensure exclusion storage exists
        if "exclude" not in self.user_preferences:
            self.user_preferences["exclude"]={
                "brands": set(),
                "categories": set(),
                "colors": set(),
                "price": None
            }

        # Detecting General exclusions (e.g., "not from fenty beauty", "except lipstick", "not pink"
        exclusion_patterns = {
        "brands": r"(?:not from|except)\s+([\w\s&]+)",  # e.g., "not from Fenty Beauty"
        "categories": r"(?:not a|not an|except)\s+([\w\s]+)",  # e.g., "not a lipstick"
        "colors": r"(?:not|except)\s+([\w\s]+)",  # e.g., "not pink"
        "price": r"(?:not|except) under\s+\$(\d+)"  # e.g., "not under $20"
    }

        for key, pattern in exclusion_patterns.items():
            match = re.findall(pattern, user_input.lower())
            if match:
                for excluded_value in match:
                    excluded_value = excluded_value.strip().lower()
                    if key == "price":
                        self.user_preferences["exclude"]["price"] = float(excluded_value)
                    else:
                        self.user_preferences["exclude"][key].add(excluded_value)

                    updated = True



        #Extract brand (if mentioned)
        if not detected_preferences["brands"]:
            for brand in self.brands:
                if brand in user_input.lower() and brand not in self.user_preferences["exclude"]["brands"]:
                    self.user_preferences["brand"] = brand
                    detected_preferences["brand"] = True
                    updated = True
                    break

        # Extract category
        if not detected_preferences["categories"]:
            for cat in self.categories:
                if cat in user_input.lower() and cat not in self.user_preferences["exclude"]["categories"]:
                    self.user_preferences["category"] = cat
                    detected_preferences["category"] = True
                    updated = True
                    break

            # Check if the user specified a general color (e.g. "pink")
        if not detected_preferences["colors"]:
            for base_color, shades in self.color_groups.items():
                if base_color in user_input.lower() and base_color not in self.user_preferences["exclude"]["colors"]:
                    self.user_preferences["color_group"] = base_color # Offer all shades of this base color
                    self.user_preferences["color"] = None # Clear specific color
                    detected_preferences["color_group"] = True
                    updated = True
                    break

        # Check if the user specified a specific color variation (e.g., "pink nude")
        if not detected_preferences["colors"]:
            for color in self.colors:
                if color in user_input.lower() and color not in self.user_preferences["exclude"]["colors"]:
                    self.user_preferences["color"] = color # Store specific shade
                    self.user_preferences["color_group"] = None # Remove general color
                    detected_preferences["color"] = True
                    updated = True
                    break

        #Extract price
        if not detected_preferences["price_range"]:
            price_pattern =  r"\$(\d+)|(\d+\s?(dollars|euros|shekels))"
            match = re.search(price_pattern, user_input)
            if match:
                self.user_preferences["price_range"] = match.group(0)
                detected_preferences["price_range"] = True
                updated = True

        #If only one preference is mentioned, reset the others:
        if sum(detected_preferences.values()) == 1:
            for key in ["brand", "category", "color", "color_group", "price_range"]:
                if not detected_preferences.get(key,False): #Reset filters not mentioned
                    self.user_preferences[key] = None # Ignore unrelated filters temporarily

        # Count the number of active filters (excluding `exclude`)
        num_active_filters = sum(1 for key in detected_preferences if self.user_preferences.get(key) is not None)

        # Ignore unrelated filters **for this request only**
        if num_active_filters == 1:
            for key in detected_preferences:
                if not detected_preferences.get(key,False):
                    self.user_preferences[key] = None  # Ignore unrelated filters **temporarily**

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
            matching_products = self.find_matching_products()

            if matching_products:
                response = "Here are some options for you:\n"
                for product in matching_products[:5]: #limit to top 5 results
                    response += f"- **{product['name']}** ({product['brand']}) - ${product['price']}\n"
                    if "url" in product:
                        response += f"  [View Product]({product['url']})\n"

                return response

            else:
                return "I couldn't find any exact matches, but let me know if you’d like other recommendations!"

        return "Can you clarify what you're looking for? I'd love to help!"

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
