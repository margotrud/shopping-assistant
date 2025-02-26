import openai
import json
import spacy
import openai
from dotenv import load_dotenv
import os
import streamlit as st
from fuzzywuzzy import process, fuzz

# Standardized color mapping
COLOR_MAPPING = {
    "red": ["ruby", "scarlet", "crimson", "wine", "brick", "cherry"],
    "pink": ["rose", "berry", "blush", "fuchsia", "bubblegum", "champagne", "nude", "pink"],
    "nude": ["beige", "sand", "taupe", "caramel", "peach", "nude"],
    "brown": ["mocha", "espresso", "chocolate", "bronze", "cocoa"],
    "purple": ["plum", "violet", "lilac", "mauve", "grape", "Amethyst"],
    "orange": ["coral", "amber", "tangerine", "apricot", "peach"],
}

class ShoppingAssistant:
    def __init__(self, product_file="data/products.json"):
        """Initialize the assistant, load product data, and set up OpenAI API."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ API Key not found. Set OPENAI_API_KEY in .env file.")

        #self.client = openai.Client(api_key=self.api_key)
        self.products = self.load_products(product_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.all_brands, self.all_categories = self.extract_attributes()
        self.previous_queries = []
        openai.api_key = self.api_key

    def load_products(self, product_file):
        """Load product data from JSON file."""
        try:
            with open(product_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("❌ Error: Product file not found.")

    def extract_attributes(self):
        """Extract unique brands and categories from products."""
        brands = {product.get("brand", "").strip().lower() for product in self.products}
        categories = {product.get("category", "").strip().lower() for product in self.products}
        return brands, categories

    def parse_user_query_with_llm(self, user_input):
        """Use OpenAI GPT to extract structured preferences from query."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                        Extract structured attributes from the user's shopping request:
                        - Identify affirmations (specific requests like 'I want a lipstick').
                        - Identify negations (e.g., 'not a foundation') and add them to exclusions.
                        - Recognize 'brand', 'category', 'color', 'price_range', and 'exclusions'.
                        - Ensure that product categories from the dataset (e.g., 'foundation', 'eyeshadow') are correctly identified.
                        - If a category or brand is clearly mentioned in the request, do NOT ask for clarification.
                        - Do NOT rely on previous queries for exclusions; only apply exclusions from the current request.
                        - If a request is vague, use previous affirmations (e.g., if only a brand is requested, remember previous categories).
                        - Ignore generic responses like 'show me all options', 'any', or similar vague terms.
                        - Always return a JSON object with these keys.
                    """},
                    {"role": "user", "content": str(user_input)}
                ],
                response_format={"type": "json_object"}
            )
            if response and response.choices:
                parsed_response = json.loads(response.choices[0].message.content) or {}
            else:
                raise ValueError("Invalid response received from OpenAI API")

            # Debug print
            print(f"🔍 Debug - Parsed Query from LLM: {parsed_response}")  # <--- Debugging Here

            # Ensure correct interpretation of categories directly from dataset
            # If category is missing, extract from "affirmations"
            if not parsed_response.get("category") and "affirmations" in parsed_response:
                if isinstance(parsed_response["affirmations"], dict):
                    print(f"⚠️ Debug - 'affirmations' is a dict. Attempting to extract category...")
                    if "category" in parsed_response["affirmations"]:
                        parsed_response["category"] = parsed_response["affirmations"]["category"]
                        print(f"✅ Debug - Extracted category from dict affirmations: {parsed_response['category']}")
                elif isinstance(parsed_response["affirmations"], list):
                    for affirmation in parsed_response["affirmations"]:
                        if isinstance(affirmation, str) and affirmation.lower() in self.all_categories:
                            parsed_response["category"] = affirmation.lower()
                            print(f"✅ Debug - Extracted category from list affirmations: {parsed_response['category']}")
                            break  # Stop at first match

                if isinstance(parsed_response["affirmations"], list):  # Handle list format
                    for affirmation in parsed_response["affirmations"]:
                        if isinstance(affirmation, str) and affirmation.lower() in self.all_categories:
                            parsed_response["category"] = affirmation.lower()
                            break  # Stop at first match

            # Convert to lowercase
            if parsed_response.get("category"):
                if isinstance(parsed_response["category"], str):
                    parsed_response["category"] = parsed_response["category"].lower()
                elif isinstance(parsed_response["category"], dict):  # Fix if LLM returns a dict
                    print(
                        f"⚠️ Debug - Expected string for 'category', but got {type(parsed_response['category'])}. Fixing...")
                    parsed_response["category"] = ""  # Reset to empty string

            if parsed_response.get("brand"):
                if isinstance(parsed_response["brand"], str):
                    parsed_response["brand"] = parsed_response["brand"].lower()
                else:
                    print(
                        f"⚠️ Debug - Expected string for 'brand', but got {type(parsed_response['brand'])}. Fixing...")
                    parsed_response["brand"] = ""  # Reset to empty string if it's not a valid string

            if parsed_response.get("brand", "") in ["all", "all options", "any"]:
                parsed_response["brand"] = None

            # If brand is missing, extract manually from user input
            if not parsed_response.get("brand") or isinstance(parsed_response["brand"], dict):  # Prevent dict error
                extracted_brand = next((brand for brand in self.all_brands if brand.lower() in user_input.lower()),
                                       None)
                if extracted_brand:
                    parsed_response["brand"] = extracted_brand
                    print(f"✅ Debug - Manually extracted brand: {parsed_response['brand']}")
                else:
                    parsed_response["brand"] = None  # Ensure brand is not a dict

            # Convert to lowercase for consistency
            if parsed_response.get("brand"):
                if isinstance(parsed_response["brand"], str):
                    parsed_response["brand"] = parsed_response["brand"].lower()
                elif isinstance(parsed_response["brand"], dict):  # Fix if LLM returns a dict
                    print(
                        f"⚠️ Debug - Expected string for 'brand', but got {type(parsed_response['brand'])}. Fixing...")
                    parsed_response["brand"] = ""  # Reset to empty string

            # Ensure brand is properly extracted if it's inside "affirmations"
            if not parsed_response.get("brand") and "affirmations" in parsed_response:
                if isinstance(parsed_response["affirmations"], dict) and "brand" in parsed_response["affirmations"]:
                    parsed_response["brand"] = parsed_response["affirmations"]["brand"]

            # Convert to lowercase for consistency
            if parsed_response.get("brand") and isinstance(parsed_response["brand"], str):
                parsed_response["brand"] = parsed_response["brand"].lower()

            print(f"✅ Debug - Extracted brand: {parsed_response.get('brand')}")  # Debugging

            # ✅ Ensure exclusions are properly extracted
            # Ensure exclusions is a properly formatted list
            # Ensure exclusions is a properly formatted list
            exclusions = parsed_response.get("exclusions", [])

            # Convert dict to list if needed
            if isinstance(exclusions, dict):
                print(f"⚠️ Debug - Expected list for 'exclusions', but got a dict. Fixing...")
                exclusions = list(exclusions.values())  # Convert dict values to list

            # Ensure exclusions is a list
            if not isinstance(exclusions, list):
                print(f"⚠️ Debug - Expected list for 'exclusions', but got {type(exclusions)}. Fixing...")
                exclusions = []

            # Convert exclusions to lowercase and remove duplicates
            exclusions = list(set(exclusion.lower() for exclusion in exclusions if isinstance(exclusion, str)))

            parsed_response["exclusions"] = exclusions

            print(f"✅ Debug - Final exclusions list: {parsed_response.get('exclusions')}")  # Debugging

            return {
                "brand": parsed_response.get("brand", ""),
                "category": parsed_response.get("category", ""),
                "color": parsed_response.get("color", ""),
                "price_range": parsed_response.get("price_range", None),
                "exclusions": exclusions
            }

        except Exception as e:
            print(f"❌ LLM Parsing Error: {e}")
            return {
                "brand": "",
                "category": "",
                "color": "",
                "price_range": None,
                "exclusions": []
            }
    def process_query(self, user_input_list):
        """Process user queries, parse with LLM, and refine recommendations based on stored preferences."""
        responses = []

        for user_input in user_input_list:
            parsed_query = self.parse_user_query_with_llm(user_input) or {}

            print(f"🔍 Debug - Parsed Query Before Processing: {parsed_query}")  # <--- Debugging Here
            selected_brand = parsed_query.get("brand", "") or st.session_state.preferences.get("brand", "")
            selected_category = parsed_query.get("category", "") or st.session_state.preferences.get("category", "")
            requested_color = parsed_query.get("color", "") or st.session_state.preferences.get("color", "")
            price_range = parsed_query.get("price_range", None)
            exclusions = parsed_query.get("exclusions", []) + st.session_state.preferences.get("exclusions", [])

            # **Step 1: Identify Filtering Priorities**
            filters = []
            if selected_brand:
                filters.append(("brand", selected_brand))
            if selected_category:
                filters.append(("category", selected_category))
            if requested_color:
                filters.append(("color", requested_color))
            if price_range:
                filters.append(("price_range", price_range))

            # **Step 2: Apply Dynamic Filtering**
            filtered_products = self.products
            print(f"🔍 Debug - Total Products Before Filtering: {len(filtered_products)}")  # Debugging

            for filter_type, filter_value in filters:
                if filter_type == "brand" and filter_value:
                    # 🚨 Skip brand filtering if exclusions exist
                    if exclusions:
                        print(f"⚠️ Skipping brand filter because exclusions exist: {exclusions}")
                    else:
                        filtered_products = [
                            product for product in filtered_products
                            if product.get("brand", "").strip().lower() == filter_value.strip().lower()
                               or fuzz.partial_ratio(product.get("brand", "").lower(), filter_value.lower()) > 80
                        ]
                        print(f"✅ Debug - Filtered {len(filtered_products)} products after brand filter")

                elif filter_type == "brand" and not filter_value:
                    print("✅ Debug - No brand filtering applied, keeping all brands")

                elif filter_type == "category":
                    # Convert both filter and dataset values to lowercase for case-insensitive comparison
                    filtered_products = [
                        product for product in filtered_products
                        if product.get("category", "").strip().lower() == filter_value.strip().lower()
                    ]

                    # If no exact matches, try fuzzy matching
                    if not filtered_products:
                        print("⚠️ No exact category matches found, trying fuzzy matching...")  # Debugging
                        filtered_products = [
                            product for product in self.products
                            if fuzz.ratio(product.get("category", "").lower(), filter_value.lower()) > 50
                            # Lowered threshold
                        ]

                    print(f"✅ Debug - Filtered {len(filtered_products)} products after category filter")




                elif filter_type == "category" and not filter_value:
                    print("✅ Debug - No category filtering applied, keeping all categories")  # Debugging

                elif filter_type == "color":
                    filtered_products = [
                        product for product in filtered_products
                        if any(fuzz.ratio(product.get("color", "").lower(), color) > 70 for color in
                               COLOR_MAPPING.get(filter_value, [filter_value]))
                    ]
                elif filter_type == "price_range":
                    filtered_products = [
                        product for product in filtered_products
                        if filter_value[0] <= product.get("price", 0) <= filter_value[1]
                    ]

            print(f"✅ Debug - Remaining Products After Filtering: {len(filtered_products)}")
            if len(filtered_products) == 0:
                print("⚠️ Debug - No products left! Check filtering conditions.")


            # **Step 3: Apply Exclusions**
            # Ensure exclusions are applied correctly
            if exclusions:
                print(f"✅ Debug - Applying exclusions: {exclusions}")  # Debugging

            filtered_products = [
                product for product in filtered_products
                if product.get("brand", "").strip().lower() not in exclusions  # Exact match, not substring match
            ]

            print(f"✅ Debug - Products after exclusions: {len(filtered_products)}")  # Debugging

            # **Step 4: Apply Scoring to the Remaining Products**
            product_scores = []
            for product in filtered_products:
                score = 0

                if selected_brand and fuzz.ratio(product.get("brand", "").lower(), selected_brand.lower()) > 85:
                    score += 2  # Prefer exact brand match

                if selected_category and fuzz.ratio(product.get("category", "").lower(),
                                                    selected_category.lower()) > 85:
                    score += 2  # Prefer exact category match

                if requested_color:
                    matched = any(fuzz.ratio(product.get("color", "").lower(), color) > 70 for color in
                                  COLOR_MAPPING.get(requested_color, [requested_color]))
                    if matched:
                        score += 3  # Color match gets high weight

                if price_range and price_range[0] <= product.get("price", 0) <= price_range[1]:
                    score += 2  # Prioritize price match

                product_scores.append((product, score))

            # **Step 5: Sort Results by Score**
            product_scores = sorted(product_scores, key=lambda x: x[1], reverse=True)

            # **Step 6: Format the Response**
            response = ""
            if product_scores:
                response += "\n🎯 Recommended Products:\n"
                for product, _ in product_scores[:5]:  # Show only the top 5 results
                    response += f"- [{product['name']}]({product['image_url']}) ({product['brand']}, {product['color']}) - ${product['price']}\n"
            else:
                response = "🤔 No exact match found. Would you like to explore other categories or brands?"

            responses.append(response)

        return "\n".join(responses)


