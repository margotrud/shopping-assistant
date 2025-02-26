# shopping-assistant
## 1. Introduction & Context
This project is about building a shopping assistant that helps users find products and make better purchase decisions. The assistant has a natural conversation with the user, asks questions, keeps track of preferences, and recommends products based on those preferences. The goal is for the assistant to feel like a real person, giving personalized suggestions based on what the user likes.
We use OpenAI's GPT-4o-mini model to power the conversation, allowing it to understand and process what the user says, improving recommendations over time.

## 2. Key Features & Workflow
### 2.1 User Profiling & Multi-turn Conversations
The assistant remembers what the user likes and updates their preferences during the conversation. For example, if the user says, "I want a red lipstick from Fenty Beauty," it saves that info (brand: Fenty Beauty, product: lipstick, color: red) to use for future suggestions.
As the conversation goes on, the assistant can ask more questions to refine its recommendations. For example, it might ask, "What’s your price range?" to help narrow down the options.

### 2.2 Product Recommendations
The assistant looks through a small dataset of products and filters them based on the user’s preferences. As it learns more about the user, the recommendations get better, always keeping their likes in mind.

### 2.3 Chat Interface
The chat interface is simple and easy to use. The user can type messages and get responses. The assistant remembers past messages, so it can keep track of the conversation.
The user will also see product suggestions with basic details like name, price, and brand, making it easy to browse products while chatting.


## 3. Tools & Technologies
OpenAI API (GPT-4o-mini): This handles the language processing, so the assistant understands what the user is saying.
Streamlit: Used for the chat interface, so users can talk to the assistant easily.
Python: I used Python to make everything work, like tracking user preferences and filtering the product recommendations.
JSON (or CSV): I stored the product data (name, category, brand, price, etc.) in a simple JSON file so it’s easy to access and filter.
## 4. Dataset
I created a small dataset with about 10-20 products. Each product has:

Name: The product's name.
Brand: The brand of the product.
Category: What type of product it is (like lipstick, foundation, etc.).
Price: How much the product costs.
Color: Available colors (for things like lipstick).
Short Description: A brief description of the product.
URL link : to the website article.
The data is stored in a JSON file to make it easy to use and filter when recommending products.

## 5. Improvement Points
While it  works, there are a few things I could improve:
Resetting Preferences for Specific Queries: If a user says, "show all options," the assistant should forget about any previous preferences (like brand or color) and show everything. Right now, if there were preferences from past messages, they’ll still apply, even when the user wants to see everything.
Handling User Input Without Filters: If the user doesn't mention anything like a brand or color preference, the assistant should show a wider range of products instead of just the ones that match past preferences. This way, the assistant can offer more variety if the user hasn’t decided yet.
UI Enhancements: The chat interface works, but adding things like product images or improving the overall look could make it a lot more fun and user-friendly.


## 6. Conclusion
The AI shopping assistant meets most of the core requirements of the assignment, but there are definitely some areas for improvement. It offers a multi-turn conversational interface and keeps track of user preferences as the conversation goes on. Using the OpenAI GPT-4o-mini model, the assistant understands user queries and tries to adjust its product recommendations as the chat continues. The Streamlit frontend provides a simple, functional chat interface, making it easy for users to interact and get product suggestions based on what they say.