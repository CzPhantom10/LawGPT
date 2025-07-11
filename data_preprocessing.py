import json
from datasets import Dataset
import os

def load_constitution_data(file_path):
    """Load and parse the Indian Constitution JSON data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_article_text(article):
    """Extract and combine all text from an article"""
    text_parts = []
    
    # Add article number and name
    if 'ArtNo' in article and 'Name' in article:
        text_parts.append(f"Article {article['ArtNo']}: {article['Name']}")
    
    # Add article description if available
    if 'ArtDesc' in article:
        text_parts.append(article['ArtDesc'])
    
    # Add clauses if available
    if 'Clauses' in article:
        for clause in article['Clauses']:
            if 'ClauseNo' in clause and 'ClauseDesc' in clause:
                text_parts.append(f"Clause {clause['ClauseNo']}: {clause['ClauseDesc']}")
            
            # Add subclauses if available
            if 'SubClauses' in clause:
                for subclause in clause['SubClauses']:
                    if 'SubClauseNo' in subclause and 'SubClauseDesc' in subclause:
                        text_parts.append(f"Subclause {subclause['SubClauseNo']}: {subclause['SubClauseDesc']}")
    
    return " ".join(text_parts)

def create_pretraining_dataset(constitution_data):
    """Create dataset for continued pretraining"""
    pretraining_data = []
    
    for section in constitution_data:
        for article in section:
            if isinstance(article, dict):
                # Skip omitted articles
                if article.get('Status') == 'Omitted':
                    continue
                
                text = extract_article_text(article)
                if text.strip():
                    pretraining_data.append({"text": text})
    
    return Dataset.from_list(pretraining_data)

def create_instruction_dataset(constitution_data):
    """Create dataset for supervised fine-tuning"""
    instruction_data = []
    
    for section in constitution_data:
        for article in section:
            if isinstance(article, dict):
                # Skip omitted articles
                if article.get('Status') == 'Omitted':
                    continue
                
                art_no = article.get('ArtNo', '')
                name = article.get('Name', '')
                
                if art_no and name:
                    # Create instruction for explaining the article
                    instruction = f"Explain Article {art_no} of the Indian Constitution"
                    output_text = extract_article_text(article)
                    
                    if output_text.strip():
                        instruction_data.append({
                            "instruction": instruction,
                            "input": "",
                            "output": output_text
                        })
                    
                    # Create instruction for summarizing the article
                    if len(output_text) > 200:  # Only for longer articles
                        instruction_summary = f"Summarize Article {art_no} of the Indian Constitution"
                        summary = f"Article {art_no} ({name}) " + (article.get('ArtDesc', output_text)[:200] + "...")
                        
                        instruction_data.append({
                            "instruction": instruction_summary,
                            "input": "",
                            "output": summary
                        })
    
    return Dataset.from_list(instruction_data)

def prepare_datasets(json_file_path):
    """Main function to prepare both datasets"""
    print("Loading constitution data...")
    constitution_data = load_constitution_data(json_file_path)
    
    print("Creating pretraining dataset...")
    pretraining_dataset = create_pretraining_dataset(constitution_data)
    
    print("Creating instruction dataset...")
    instruction_dataset = create_instruction_dataset(constitution_data)
    
    print(f"Pretraining dataset size: {len(pretraining_dataset)}")
    print(f"Instruction dataset size: {len(instruction_dataset)}")
    
    return pretraining_dataset, instruction_dataset

if __name__ == "__main__":
    # Test the data preprocessing
    pretraining_ds, instruction_ds = prepare_datasets("COI.json")
    
    print("\nSample pretraining data:")
    print(pretraining_ds[0]['text'][:200] + "...")
    
    print("\nSample instruction data:")
    print(f"Instruction: {instruction_ds[0]['instruction']}")
    print(f"Output: {instruction_ds[0]['output'][:200]}...")
