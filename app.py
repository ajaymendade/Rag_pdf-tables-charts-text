from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
from pdf_processor import PDFProcessor  
from io import BytesIO
import fitz  
import uuid
from openai import OpenAI
import os
import json

app = Flask(__name__)
pdf_processor = PDFProcessor()

openai_key = OpenAI(
    api_key='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'
)
# Setup MongoDB client
client = MongoClient('mongodb+srv://ragpdf:ragpdf@database.u3tj9jn.mongodb.net/')
db = client.database 
texts_collection = db.page_data
images_collection = db.images
tables_collection = db.tables

def save_vector_to_mongodb(collection, data):
    """Utility function to save vector data to the specified MongoDB collection."""
    collection.insert_one(data)

def similarity_search(prompt_vector):
    # Assuming 'db.page_vectors' is your collection of page vectors
    similar_pages = db.page_data.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector",
                "queryVector": prompt_vector,
                "numCandidates": 50,
                "limit": 5
            }
        }
    ])
    
    # Convert the cursor to a list and return
    return list(similar_pages)

def parse_gpt_response(chat_response):
    # Initialize containers for different types of queries
    text_queries, chart_queries, table_queries = [], [], []
    
    # Split the response into sections based on predefined labels
    sections = chat_response.split("QUERIES:")
    for section in sections[1:]:  # Skip the first split part, which is before the first label
        title, queries = section.strip().split(":", 1)
        queries = queries.strip().split("\n")  # Assuming each query is on a new line
        
        if "TEXT" in title:
            text_queries.extend(queries)
        elif "CHART" in title:
            chart_queries.extend(queries)
        elif "TABLE" in title:
            table_queries.extend(queries)
    
    return text_queries, chart_queries, table_queries



@app.route('/')
def home():
    return render_template('upload.html')



@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        print(prompt)

        # Your existing code for processing the prompt and generating a GPT response
        # For demonstration, this part is simplified
        prompt_vector = pdf_processor.vectorize_text(prompt)
        similar_pages = similarity_search(prompt_vector)

        compiled_data_for_gpt = "PROMPT:\n" + prompt + "\n\nRELATED DATA:\n"
        for page_info in similar_pages:
            unique_key = page_info['unique_key']
            page_number = page_info['page']

            page_text_data = texts_collection.find_one({"unique_key": unique_key, "page": page_number})
            if page_text_data:
                compiled_data_for_gpt += "TEXT:\n" + page_text_data['original_text'] + "\n"

            image_count = images_collection.count_documents({"unique_key": unique_key, "page": page_number})
            if image_count > 0:
                compiled_data_for_gpt += "\n[IMAGES included on this page.]\n"
            table_count = tables_collection.count_documents({"unique_key": unique_key, "page": page_number})
            if table_count > 0:
                compiled_data_for_gpt += "\n[TABLES included on this page.]\n"

        system_message_content = """
        Generate a structured JSON response that precisely conforms to the requirements for displaying text, tables, and charts on a web interface. The response should contain three distinct sections: "text", "tables", and "charts". Each section must adhere to the following specifications:

            1. "text": A plain text string or an array of strings providing relevant narrative or explanations.

            2. "tables": An array of objects, where each object represents a table. Each table object should include:
            - "title": A string indicating the title of the table.
            - "data": A two-dimensional array where the first sub-array contains the column headers, and the subsequent sub-arrays contain rows of data.
            - "options": An object containing DataTables configuration options such as paging, searching, and ordering. Follow the DataTables library structure.

            3. "charts": An array of objects, where each object represents a chart configuration compatible with the Chart.js library. Each chart object should include:
            - "type": A string indicating the chart type (e.g., "bar", "line").
            - "data": An object containing:
                - "labels": An array of strings for the chart labels.
                - "datasets": An array of objects, each representing a dataset with properties such as "label", "data" (an array of data points), "backgroundColor", and "borderColor".
            - "options": An object with Chart.js configuration options.

            If the input data is incomplete or if there are gaps in the provided information, use your capabilities to intelligently generate or infer missing details to complete the response. The generated data should be plausible and relevant to the context of the provided input.

            The structured JSON response should strictly contain only these three sections and nothing else outside the specified format. This structured output is intended for direct use in a web application that utilizes DataTables and Chart.js to display information dynamically based on the provided input.

            Example input: Data regarding the Indian economy is provided, with a prompt asking to compare the economies of India and the United States. If data for the US economy is not included in the input, intelligently generate or infer relevant data to facilitate the comparison.

        """


        messages = [
            {
                "role": "system",
                "content": system_message_content
            },
            {
                "role": "user",
                "content": compiled_data_for_gpt
            }
        ]
        response = openai_key.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )
        
        response_content = response.choices[0].message.content.strip().strip('```json')
        parsed_data = json.loads(response_content)
        return render_template('chatbot.html', data = parsed_data)
    else:
        return render_template('chatbot.html', data = None)
    
    
                
@app.route('/upload', methods=['POST'])
def upload_pdfs():
    # Check if 'pdf' is present in the uploaded files
    if 'pdf' not in request.files:
        return 'No file part', 400

    files = request.files.getlist('pdf')  # Get a list of files from the 'pdf' form name

    # Ensure there's at least one file selected
    if not files or files[0].filename == '':
        return 'No selected file', 400

    for file in files:
        # You may want to check each file's extension here as well
        if file:  # If there's a file and it's not an empty string
            unique_key = str(uuid.uuid4())  # Generate a unique key for this PDF upload
            pdf_stream = BytesIO(file.read())

            processed_data = pdf_processor.process_pdf(pdf_stream, unique_key)

            # Check if processed_data is None
            if processed_data is None:
                return 'Error processing PDF', 500

            # Iterate through processed data and save to MongoDB, including the unique_key
            if 'page_data' in processed_data:
                for item in processed_data['page_data']:
                    texts_collection.insert_one(item)
            if 'images_data' in processed_data:
                for item in processed_data['images_data']:
                    images_collection.insert_one(item)
            if 'tables_data' in processed_data:
                for item in processed_data['tables_data']:
                    tables_collection.insert_one(item)
            print(unique_key)
    # After all files are processed
    return render_template('upload.html', message='PDFs processed and data saved to MongoDB')



if __name__ == '__main__':
    app.run(debug=True)

