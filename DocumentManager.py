from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from dotenv import load_dotenv
import os
import glob
from tqdm import tqdm

load_dotenv()

class DocumentManager:
    def __init__(self, directory_path=None, glob_pattern="**/*.md", recursive=True):
        # Use environment variable if directory_path is not provided
        if directory_path is None:
            directory_path = os.getenv("DOCUMENTS_DIRECTORY", "./marckdown_folder")
        
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self.documents = []
        self.all_sections = []
    
    def _get_file_paths(self):
        """Get all file paths matching the pattern, recursively if enabled."""
        if self.recursive:
            # Use recursive glob pattern
            pattern = os.path.join(self.directory_path, self.glob_pattern)
        else:
            # Use non-recursive pattern
            pattern = os.path.join(self.directory_path, "*.md")
        
        return glob.glob(pattern, recursive=self.recursive)
    
    def load_documents(self):
        """Load documents one at a time to be memory-efficient."""
        file_paths = self._get_file_paths()
        
        if not file_paths:
            print(f"No markdown files found in {self.directory_path}")
            return
        
        print(f"Found {len(file_paths)} markdown files to process...")
        
        # Process each file individually to save memory
        for file_path in tqdm(file_paths, desc="Loading documents"):
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                doc = loader.load()
                if doc:  # Only add if document was successfully loaded
                    self.documents.extend(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.documents)} documents")

    def split_documents(self):
        """Split documents into sections with progress tracking."""
        if not self.documents:
            print("No documents to split")
            return
            
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        print(f"Splitting {len(self.documents)} documents into sections...")
        
        for doc in tqdm(self.documents, desc="Splitting documents"):
            try:
                sections = text_splitter.split_text(doc.page_content)
                self.all_sections.extend(sections)
            except Exception as e:
                print(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {e}")
                continue
        
        print(f"Created {len(self.all_sections)} sections from {len(self.documents)} documents")