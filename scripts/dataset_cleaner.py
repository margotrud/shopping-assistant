import json

class DatasetCleaner:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = self.load_data()

    def load_data(self):
        """
        Loads the dataset from JSON file.
        :return:
        """
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {self.input_file} not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON in {self.input_file}.")
            return []

    def save_data(self):
        """
        Saves the cleaned dataset to JSON.
        :return:
        """
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        print(f"Cleaned dataset saved to {self.output_file}")

    def clean_data(self):
        """
        Cleans dataset by handling missing values and formatting.
        :return:
        """

        for product in self.data:
            # Set default values for missing fields
            product.setdefault("id", None)
            product.setdefault("name", "Unknown product")
            product.setdefault("category", "Unknown Category")
            product.setdefault("brand", "Unknown Brand")
            product.setdefault("price", None)
            product.setdefault("color", "Unknown Color")
            product.setdefault("description", "")

            # Ensure category, brand, and color are properly formatted
            product["category"] = product["category"].strip().title() if product["category"] else "Unknown Category"
            product["brand"] = product["brand"].strip().title() if product["brand"] else "Unknown Brand"
            product["color"] = product["color"].strip().title() if product["color"] else "Unknown Color"


    def process(self):
        """
        Runs the full dataset cleaning pipeline.
        :return:
        """
        print("Cleaning dataset...")
        self.clean_data()
        self.save_data()
        print('Dataset cleaning complete.')


# Run the script:
if __name__ == '__main__':
    input_path = "../data/products.json"  # Adjust path if needed
    output_path = "../data/products_cleaned.json"

    cleaner = DatasetCleaner(input_path, output_path)
    cleaner.process()
