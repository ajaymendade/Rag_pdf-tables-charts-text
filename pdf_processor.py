import torch
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import io
import fitz

class PDFProcessor:
    def __init__(self):
        # Initialize text vectorization model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize image vectorization model
        self.image_model = resnet50(pretrained=True)
        self.image_model.eval()  # Set to evaluation mode
        
        # Define image transformations
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def vectorize_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()

    def vectorize_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            vector = self.image_model(image_tensor)
        return torch.nn.functional.normalize(vector, p=2, dim=1).squeeze().tolist()


    def cluster_text_blocks_into_tables(self, text_blocks):
        if not text_blocks:
            return []

        # Sort text blocks by their top Y coordinate to group them by rows
        text_blocks = sorted(text_blocks, key=lambda block: block[1])

        tables = []  # This will store the final tables as lists of strings (rows)
        current_row = []
        last_y0 = text_blocks[0][1]  # Initialize with the Y coordinate of the first block

        for block in text_blocks:
            # Corrected unpacking to match the structure of blocks returned by PyMuPDF
            x0, y0, x1, y1, text, _, _ = block  # Add placeholders for the extra elements
            if abs(y0 - last_y0) < 10:  # 10 is a threshold for row height difference
                current_row.append(text)
            else:
                if current_row:  # New row detected, add the previous row to tables
                    tables.append(" ".join(current_row))
                    current_row = [text]  # Start a new row
                last_y0 = y0
        if current_row:  # Add the last row to tables
            tables.append(" ".join(current_row))

        return tables

    def process_pdf(self, pdf_stream, unique_key):
        try:
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            # Initialize dictionaries to store processed data
            page_data = []
            images_data = []
            tables_data = []

            for page_num, page in enumerate(doc):
                # Vectorize the entire page text
                full_page_text = page.get_text("text")
                page_vector = self.vectorize_text(full_page_text)

                # Store vectorized page text with metadata
                page_data.append({
                    'unique_key': unique_key,
                    'page': page_num,
                    'vector': page_vector,
                    'original_text': full_page_text  # Saving original full page text
                })

                # Process and store image data
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_vector = self.vectorize_image(base_image["image"])
                    images_data.append({
                        'unique_key': unique_key,
                        'page': page_num,
                        'vector': image_vector,
                        'description': f"Image_{img_index}_Page_{page_num}"  # Example description
                    })

                # Process and store table data (assuming method to extract tables)
                # Example placeholders for table extraction
                table_texts = self.cluster_text_blocks_into_tables(page.get_text("blocks"))
                for table_text in table_texts:
                    table_vector = self.vectorize_text(table_text)
                    tables_data.append({
                        'unique_key': unique_key,
                        'page': page_num,
                        'vector': table_vector,
                        'original_text': table_text  # Saving original table text
                    })

            return {
                'page_data': page_data,
                'images_data': images_data,
                'tables_data': tables_data
            }

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None


              
