import json
import spacy
import openai
from dotenv import load_dotenv
import os

from fuzzywuzzy import process

# Standardized color mapping (can be expanded)
COLOR_MAPPING = {
    "red": ["ruby", "scarlet", "crimson", "wine", "brick", "cherry"],
    "pink": ["rose", "berry", "blush", "fuchsia", "bubblegum", "champagne"],
    "nude": ["beige", "sand", "taupe", "caramel", "peach", "nude"],
    "brown": ["mocha", "espresso", "chocolate", "bronze", "cocoa"],
    "purple": ["plum", "violet", "lilac", "mauve", "grape"],
    "orange": ["coral", "amber", "tangerine", "apricot"],
}


class ShoppingAssistant:
    def __init__(self, product_file="data/products.json"):
        """Initialize the assistant, load product data, and set up OpenAI API."""
        load_dotenv()  # This will load the .env file
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ API Key not found. Set OPENAI_API_KEY in .env file.")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.products = self.load_products(product_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.all_brands, self.all_categories = self.extract_attributes()
    def load_products(self, product_file):
        """Load product data from JSON file."""
        try:
            with open(product_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("❌ Error: Product file not found.")

    def extract_attributes(self):
        """Extract unique brands and categories from products."""
        brands = {product["brand"].strip().lower() for product in self.products}
        categories = {product["category"].strip().lower() for product in self.products}

        return brands, categories

    def parse_user_query_with_llm(self, user_input):
        """Use OpenAI GPT to extract structured preferences from query."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                        Extract structured attributes (brand, category, exclusions) from the user's shopping request.
                        - If the user specifies a color (e.g., 'red lipstick'), always include it under 'color'.
                        - If no color is mentioned, return 'color': None.
                        - Always return a JSON object with 'brand', 'category', 'color', 'exclusions'.
                    """},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ LLM Parsing Error: {e}")
            return {"brand": "", "category": "", "color": None, "exclusions": []}  # Default fallback
    def process_query(self, user_input_list):
        """Process user queries, parse with LLM, and filter products from dataset."""
        for user_input in user_input_list:
            parsed_query = self.parse_user_query_with_llm(user_input)

            if not parsed_query or not isinstance(parsed_query, dict):
                print(f"❌ Error: No valid response for query: '{user_input}'\n")
                continue  # Skip to next query

            selected_brand = parsed_query.get("brand", "")
            selected_brand = selected_brand.lower() if isinstance(selected_brand, str) else None

            selected_category = parsed_query.get("category")
            selected_category = selected_category.lower() if isinstance(selected_category, str) else None

            requested_color = parsed_query.get("color")
            requested_color = requested_color.lower() if isinstance(requested_color, str) else None

            excluded_brands = parsed_query.get("exclusions")  # ✅ Ensure it's a list
            excluded_brands = excluded_brands if isinstance(excluded_brands,
                                                            list) else []  # ✅ If None, replace with an empty list
            excluded_brands = [b.lower() for b in excluded_brands if isinstance(b, str)]

            standard_color = self.match_standard_color(requested_color)

            # Ensure exclusions override brand selection
            selected_brand = None if excluded_brands else selected_brand
            selected_brand = selected_brand if selected_brand in self.all_brands else None
            selected_category = selected_category if selected_category in self.all_categories else None

            # Filter products based on extracted preferences
            matching_products = [
                product for product in self.products
                if (not selected_brand or selected_brand == product["brand"].lower())  # Match brand if specified
                   and (not selected_category or selected_category == product["category"].lower())  # Match category
                   and all(excluded not in product["brand"].lower() for excluded in excluded_brands)  # Exclude brands
                   and (not standard_color or any(
                    color in product["color"].lower() for color in COLOR_MAPPING[standard_color]))
            ]

            if matching_products:
                response = "\n🎯 Recommended Products:\n"
                for product in matching_products:
                    response += f"- {product['name']} ({product['brand']}) - {product['price']} USD\n"
            else:
                response = "❌ No matching products found in the dataset."

            return response  # ✅ Now returning recommendations

    def match_standard_color(self, requested_color):
        """Finds the closest standard color for user requests."""
        if not requested_color:
            return None  # No color requested

        requested_color = requested_color.lower()

        # Find the best match from standard colors
        best_match = process.extractOne(requested_color, COLOR_MAPPING.keys(), score_cutoff=75)

        return best_match[0] if best_match else None
