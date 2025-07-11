"""
Enhanced data preprocessing for multiple legal datasets
Handles: Constitution (COI), IPC, CPC, NIA, and IndicLegalQA datasets
"""
import json
from datasets import Dataset, concatenate_datasets
import os

def load_json_data(file_path):
    """Load and parse JSON data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_constitution_data(data):
    """Process Indian Constitution data (COI.json)"""
    pretraining_data = []
    instruction_data = []
    
    for section in data:
        for article in section:
            if isinstance(article, dict) and article.get('Status') != 'Omitted':
                art_no = article.get('ArtNo', '')
                name = article.get('Name', '')
                
                # Extract full text
                text_parts = []
                if art_no and name:
                    text_parts.append(f"Article {art_no}: {name}")
                
                if 'ArtDesc' in article:
                    text_parts.append(article['ArtDesc'])
                
                if 'Clauses' in article:
                    for clause in article['Clauses']:
                        if 'ClauseNo' in clause and 'ClauseDesc' in clause:
                            text_parts.append(f"Clause {clause['ClauseNo']}: {clause['ClauseDesc']}")
                        
                        if 'SubClauses' in clause:
                            for subclause in clause['SubClauses']:
                                if 'SubClauseNo' in subclause and 'SubClauseDesc' in subclause:
                                    text_parts.append(f"Subclause {subclause['SubClauseNo']}: {subclause['SubClauseDesc']}")
                
                full_text = " ".join(text_parts)
                if full_text.strip():
                    # For pretraining
                    pretraining_data.append({"text": full_text})
                    
                    # For instruction tuning
                    if art_no and name:
                        instruction_data.append({
                            "instruction": f"Explain Article {art_no} of the Indian Constitution",
                            "input": "",
                            "output": full_text
                        })
    
    return pretraining_data, instruction_data

def process_ipc_data(data):
    """Process Indian Penal Code data (ipc.json)"""
    pretraining_data = []
    instruction_data = []
    
    for section in data:
        if isinstance(section, dict):
            section_num = section.get('Section', '')
            title = section.get('title', '')
            description = section.get('description', '')
            chapter_title = section.get('chapter_title', '')
            
            # Build section text
            text_parts = []
            if section_num:
                text_parts.append(f"Section {section_num} IPC")
            if title:
                text_parts.append(f"Title: {title}")
            if chapter_title:
                text_parts.append(f"Chapter: {chapter_title}")
            if description:
                text_parts.append(f"Description: {description}")
            
            full_text = " ".join(text_parts)
            if full_text.strip():
                # For pretraining
                pretraining_data.append({"text": full_text})
                
                # For instruction tuning
                if section_num and title:
                    instruction_data.append({
                        "instruction": f"Explain Section {section_num} of the Indian Penal Code",
                        "input": "",
                        "output": full_text
                    })
                    
                    if description:
                        instruction_data.append({
                            "instruction": f"What does Section {section_num} IPC say about {title.lower()}?",
                            "input": "",
                            "output": f"Section {section_num} IPC ({title}): {description}"
                        })
    
    return pretraining_data, instruction_data

def process_cpc_data(data):
    """Process Code of Civil Procedure data (cpc.json)"""
    pretraining_data = []
    instruction_data = []
    
    for section in data:
        if isinstance(section, dict):
            section_num = section.get('section', '')
            title = section.get('title', '')
            description = section.get('description', '')
            
            # Build section text
            text_parts = []
            if section_num:
                text_parts.append(f"Section {section_num} CPC")
            if title:
                text_parts.append(f"Title: {title}")
            if description:
                text_parts.append(f"Description: {description}")
            
            full_text = " ".join(text_parts)
            if full_text.strip():
                # For pretraining
                pretraining_data.append({"text": full_text})
                
                # For instruction tuning
                if section_num and title:
                    instruction_data.append({
                        "instruction": f"Explain Section {section_num} of the Code of Civil Procedure",
                        "input": "",
                        "output": full_text
                    })
    
    return pretraining_data, instruction_data

def process_legal_qa_data(data):
    """Process IndicLegalQA dataset"""
    instruction_data = []
    
    for item in data:
        if isinstance(item, dict):
            question = item.get('question', '')
            answer = item.get('answer', '')
            case_name = item.get('case_name', '')
            judgement_date = item.get('judgement_date', '')
            
            if question and answer:
                # Add case context to the answer if available
                enhanced_answer = answer
                if case_name:
                    enhanced_answer = f"In the case of {case_name}, {answer}"
                if judgement_date:
                    enhanced_answer += f" (Judgement dated: {judgement_date})"
                
                instruction_data.append({
                    "instruction": question,
                    "input": "",
                    "output": enhanced_answer
                })
    
    return [], instruction_data  # No pretraining data from Q&A

def process_nia_data(data):
    """Process NIA data (nia.json) - structure to be determined"""
    # Check the structure first
    pretraining_data = []
    instruction_data = []
    
    if isinstance(data, list) and len(data) > 0:
        sample = data[0]
        print(f"NIA data structure: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
        
        # Add processing logic based on actual structure
        for item in data:
            if isinstance(item, dict):
                # Generic processing - adapt based on actual structure
                text_parts = []
                for key, value in item.items():
                    if isinstance(value, str) and value.strip():
                        text_parts.append(f"{key}: {value}")
                
                if text_parts:
                    full_text = " ".join(text_parts)
                    pretraining_data.append({"text": full_text})
    
    return pretraining_data, instruction_data

def prepare_enhanced_datasets():
    """Prepare datasets from all available legal sources"""
    print("🔍 Loading all legal datasets...")
    
    all_pretraining_data = []
    all_instruction_data = []
    
    # Process Constitution data
    if os.path.exists("COI.json"):
        print("📜 Processing Constitution of India...")
        coi_data = load_json_data("COI.json")
        coi_pretrain, coi_instruct = process_constitution_data(coi_data)
        all_pretraining_data.extend(coi_pretrain)
        all_instruction_data.extend(coi_instruct)
        print(f"   ✅ Constitution: {len(coi_pretrain)} pretraining, {len(coi_instruct)} instruction samples")
    
    # Process IPC data
    if os.path.exists("ipc.json"):
        print("⚖️ Processing Indian Penal Code...")
        ipc_data = load_json_data("ipc.json")
        ipc_pretrain, ipc_instruct = process_ipc_data(ipc_data)
        all_pretraining_data.extend(ipc_pretrain)
        all_instruction_data.extend(ipc_instruct)
        print(f"   ✅ IPC: {len(ipc_pretrain)} pretraining, {len(ipc_instruct)} instruction samples")
    
    # Process CPC data
    if os.path.exists("cpc.json"):
        print("📋 Processing Code of Civil Procedure...")
        cpc_data = load_json_data("cpc.json")
        cpc_pretrain, cpc_instruct = process_cpc_data(cpc_data)
        all_pretraining_data.extend(cpc_pretrain)
        all_instruction_data.extend(cpc_instruct)
        print(f"   ✅ CPC: {len(cpc_pretrain)} pretraining, {len(cpc_instruct)} instruction samples")
    
    # Process Legal Q&A data
    if os.path.exists("IndicLegalQA Dataset_10K_Revised.json"):
        print("❓ Processing IndicLegalQA Dataset...")
        qa_data = load_json_data("IndicLegalQA Dataset_10K_Revised.json")
        qa_pretrain, qa_instruct = process_legal_qa_data(qa_data)
        all_instruction_data.extend(qa_instruct)
        print(f"   ✅ LegalQA: {len(qa_instruct)} instruction samples")
    
    # Process NIA data
    if os.path.exists("nia.json"):
        print("🔍 Processing NIA data...")
        try:
            nia_data = load_json_data("nia.json")
            nia_pretrain, nia_instruct = process_nia_data(nia_data)
            all_pretraining_data.extend(nia_pretrain)
            all_instruction_data.extend(nia_instruct)
            print(f"   ✅ NIA: {len(nia_pretrain)} pretraining, {len(nia_instruct)} instruction samples")
        except Exception as e:
            print(f"   ⚠️ NIA processing failed: {str(e)}")
    
    # Create datasets
    print(f"\n📊 Total dataset sizes:")
    print(f"   🔄 Pretraining: {len(all_pretraining_data)} samples")
    print(f"   🎯 Instruction: {len(all_instruction_data)} samples")
    
    pretraining_dataset = Dataset.from_list(all_pretraining_data) if all_pretraining_data else None
    instruction_dataset = Dataset.from_list(all_instruction_data) if all_instruction_data else None
    
    return pretraining_dataset, instruction_dataset

if __name__ == "__main__":
    # Test the enhanced preprocessing
    pretraining_ds, instruction_ds = prepare_enhanced_datasets()
    
    if pretraining_ds:
        print(f"\n📝 Sample pretraining data:")
        print(pretraining_ds[0]['text'][:200] + "...")
    
    if instruction_ds:
        print(f"\n💬 Sample instruction data:")
        print(f"Instruction: {instruction_ds[0]['instruction']}")
        print(f"Output: {instruction_ds[0]['output'][:200]}...")
