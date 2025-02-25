import json
import spacy
import openai
from dotenv import load_dotenv
import os
import streamlit as st

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
                        - Always return a JSON object with 'brand', 'category', 'color', 'exclusions'.
                    """},
                    {"role": "user", "content": str(user_input)}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ LLM Parsing Error: {e}")
            return {"brand": "", "category": "", "color": "", "exclusions": []}  # Default fallback

    def process_query(self, user_input_list):
        """Process user queries, parse with LLM, and filter products from dataset."""
        responses = []  # Collect all responses instead of returning early

        for user_input in user_input_list:
            parsed_query = self.parse_user_query_with_llm(user_input)

            if not parsed_query or not isinstance(parsed_query, dict):
                print(f"❌ Error: No valid response for query: '{user_input}'\n")
                responses.append("❌ No valid response.")
                continue  # Skip to next query

            # Retrieve stored preferences from session state (default to 'any' for no filter)
            selected_brand = parsed_query.get("brand") or st.session_state.preferences.get("brand")
            selected_category = parsed_query.get("category") or st.session_state.preferences.get("category") or "any"
            requested_color = parsed_query.get("color") or st.session_state.preferences.get("color") or "any"
            excluded_brands = parsed_query.get("exclusions") or st.session_state.preferences.get("exclusions", [])

            # Ensure exclusions are always a list and lowercase for matching
            excluded_brands = excluded_brands if isinstance(excluded_brands, list) else []
            excluded_brands = [b.lower() for b in excluded_brands if isinstance(b, str)]

            # If category or color is not specified, don't apply any filter for them
            if not selected_category:
                selected_category = "any"  # Treat as no filter
            if not requested_color:
                requested_color = "any"  # Treat as no filter

            # Match user-requested color to standard color
            standard_color = self.match_standard_color(requested_color)

            # Ensure exclusions override brand selection
            selected_brand = None if excluded_brands else selected_brand
            selected_brand = selected_brand if selected_brand and selected_brand.lower() in self.all_brands else None
            selected_category = selected_category if selected_category != "any" and selected_category in self.all_categories else "any"
            selected_category = selected_category if selected_category != "any" else None  # "any" means no category filter

            # Filter products based on stored preferences (brand, category, exclusions, color)
            matching_products = [
                product for product in self.products
                if
                (not selected_brand or selected_brand.lower() == product["brand"].lower())  # Match brand if specified
                and (not selected_category or selected_category == "any" or selected_category.lower() == product[
                    "category"].lower())  # Match category or all
                and all(excluded not in product["brand"].lower() for excluded in excluded_brands)  # Exclude brands
                and (not standard_color or any(  # Match color (e.g., pink)
                    color in product["color"].lower() for color in COLOR_MAPPING.get(standard_color, [])))
            ]

            if matching_products:
                response = "\n🎯 Recommended Products:\n"
                for product in matching_products:
                    response += f"- {product['name']} ({product['brand']}, {product['color']}) - {product['price']} USD\n"
            else:
                response = "❌ No matching products found in the dataset."

            responses.append(response)  # Store response instead of returning early

        return "\n".join(responses)  # Return all responses instead of stopping early

    def match_standard_color(self, requested_color):
        """Finds the closest standard color for user requests."""
        if not requested_color:
            return None  # No color requested

        requested_color = requested_color.lower()

        # Find the best match from standard colors
        best_match = process.extractOne(requested_color, COLOR_MAPPING.keys(), score_cutoff=75)

        return best_match[0] if best_match else None
