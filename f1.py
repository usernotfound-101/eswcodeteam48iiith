#!/usr/bin/env python3
"""
SQuAD F1 Evaluation Script for Qualcomm Phi Model via ADB
Uses official SQuAD dataset and evaluation metrics
"""

import subprocess
import re
import time
import json
from typing import List, Dict
from datasets import load_dataset
import evaluate

class PhiSQuADEvaluator:
    def __init__(self, device_path="/data/local/tmp/genie_bundle"):
        self.device_path = device_path
        self.squad_metric = evaluate.load("squad")
        
    def run_adb_inference(self, prompt: str, timeout: int = 30) -> str:
        """
        Run inference on the Phi model via ADB
        """
        # Escape quotes and special characters in the prompt
        escaped_prompt = prompt.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
        
        # Complete ADB command
        adb_command = [
            'adb', 'shell', 
            f'cd {self.device_path} && '
            f'export LD_LIBRARY_PATH=$PWD && '
            f'export ADSP_LIBRARY_PATH=$PWD && '
            f'./genie-t2t-run -c genie_config.json -p "{escaped_prompt}"'
        ]
        
        try:
            print(f"Running inference...")
            result = subprocess.run(
                adb_command, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                shell=False
            )
            
            if result.returncode != 0:
                print(f"Error running ADB command: {result.stderr}")
                return ""
            
            # Extract output between [BEGIN] and [END] tags
            output = self.extract_model_output(result.stdout)
            return output
            
        except subprocess.TimeoutExpired:
            print("ADB command timed out")
            return ""
        except Exception as e:
            print(f"Error running ADB command: {e}")
            return ""
    
    def extract_model_output(self, raw_output: str) -> str:
        """
        Extract the actual model output from the raw ADB response
        """
        # Look for text between [BEGIN]: and [END]
        begin_pattern = r'\[BEGIN\]:\s*(.*?)\s*\[END\]'
        match = re.search(begin_pattern, raw_output, re.DOTALL)
        
        if match:
            output = match.group(1).strip()
            # Clean up any extra whitespace and join words that got separated
            output = re.sub(r'\s+', ' ', output)
            return output
        
        # Fallback: try to find content after assistant token
        assistant_pattern = r'<\|assistant\|\>\s*(.*?)(?:\[|$)'
        match = re.search(assistant_pattern, raw_output, re.DOTALL)
        
        if match:
            output = match.group(1).strip()
            # Remove any remaining system tags
            output = re.sub(r'\[.*?\]', '', output).strip()
            output = re.sub(r'\s+', ' ', output)
            return output
        
        # Last resort: return cleaned raw output
        cleaned = re.sub(r'\[.*?\]', '', raw_output).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned
    
    def create_phi_prompt(self, context: str, question: str) -> str:
        """
        Create a properly formatted prompt for the Phi model with SQuAD-style QA
        """
        system_msg = "You are a helpful assistant. Answer the question based on the given context. Be concise and accurate, try to answer in a single line."
        prompt = f'<|system|>\n{system_msg}<|end|>\n<|user|>\nContext: {context}\n\nQuestion: {question}<|end|>\n<|assistant|>\n'
        return prompt
    
    def evaluate_squad_subset(self, num_samples: int = 50, split: str = "validation") -> Dict:
        """
        Evaluate the model on a subset of SQuAD dataset
        """
        # Load SQuAD dataset
        print("Loading SQuAD dataset...")
        squad_dataset = load_dataset("squad")
        
        # Get subset of validation data
        val_data = squad_dataset[split].select(range(min(num_samples, len(squad_dataset[split]))))
        
        predictions = []
        references = []
        
        print(f"Evaluating on {len(val_data)} SQuAD samples...")
        
        for i, example in enumerate(val_data):
            context = example['context']
            question = example['question']
            # SQuAD answers are in format {'text': [answer], 'answer_start': [start_pos]}
            true_answers = example['answers']['text']
            example_id = example['id']
            
            print(f"\n--- Sample {i+1}/{len(val_data)} (ID: {example_id}) ---")
            print(f"Question: {question[:100]}...")
            
            # Create prompt and run inference
            prompt = self.create_phi_prompt(context, question)
            predicted_answer = self.run_adb_inference(prompt)
            
            if not predicted_answer:
                print("No output received, using empty string...")
                predicted_answer = ""
            
            print(f"True answers: {true_answers}")
            print(f"Predicted: {predicted_answer}")
            
            # Format for SQuAD metric evaluation
            predictions.append({
                'id': example_id,
                'prediction_text': predicted_answer
            })
            
            references.append({
                'id': example_id,
                'answers': example['answers']
            })
            
            # Add delay to avoid overwhelming the device
            time.sleep(2)
        
        # Calculate SQuAD metrics (F1 and Exact Match)
        print("\nCalculating SQuAD metrics...")
        results = self.squad_metric.compute(predictions=predictions, references=references)
        
        # Add additional info
        results.update({
            'num_samples': len(predictions),
            'predictions': predictions,
            'references': references
        })
        
        return results
    
    def evaluate_custom_qa(self, qa_pairs: List[Dict]) -> Dict:
        """
        Evaluate on custom QA pairs using SQuAD format
        """
        predictions = []
        references = []
        
        print(f"Evaluating on {len(qa_pairs)} custom QA pairs...")
        
        for i, qa in enumerate(qa_pairs):
            context = qa.get('context', '')
            question = qa['question']
            true_answers = qa['answers'] if isinstance(qa['answers'], list) else [qa['answers']]
            qa_id = qa.get('id', f"custom_{i}")
            
            print(f"\n--- Custom Sample {i+1}/{len(qa_pairs)} ---")
            print(f"Question: {question}")
            
            # Create prompt and run inference
            if context:
                prompt = self.create_phi_prompt(context, question)
            else:
                # For questions without context, use simpler format
                system_msg = "You are a helpful assistant. Answer the question concisely and accurately."
                prompt = f'<|system|>\n{system_msg}<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n'
            
            predicted_answer = self.run_adb_inference(prompt)
            
            if not predicted_answer:
                predicted_answer = ""
            
            print(f"True answers: {true_answers}")
            print(f"Predicted: {predicted_answer}")
            
            # Format for SQuAD metric evaluation
            predictions.append({
                'id': qa_id,
                'prediction_text': predicted_answer
            })
            
            references.append({
                'id': qa_id,
                'answers': {
                    'text': true_answers,
                    'answer_start': [0] * len(true_answers)  # Dummy start positions
                }
            })
            
            time.sleep(1)
        
        # Calculate SQuAD metrics
        results = self.squad_metric.compute(predictions=predictions, references=references)
        results.update({
            'num_samples': len(predictions),
            'predictions': predictions,
            'references': references
        })
        
        return results

def create_sample_qa_pairs() -> List[Dict]:
    """
    Create sample QA pairs for testing
    """
    return [
        {
            "id": "test_1",
            "context": "Paris is the capital and most populous city of France. It is located in the north-central part of the country.",
            "question": "What is the capital of France?",
            "answers": ["Paris"]
        },
        {
            "id": "test_2",
            "context": "Droupadi Murmu is the 15th and current President of India. She assumed office on 25 July 2022.",
            "question": "Who is the current President of India?",
            "answers": ["Droupadi Murmu"]
        },
        {
            "id": "test_3",
            "context": "",  # No context - general knowledge question
            "question": "What is 15 + 27?",
            "answers": ["42"]
        },
        {
            "id": "test_4",
            "context": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than two and a half times that of all other planets combined.",
            "question": "Which is the largest planet in our solar system?",
            "answers": ["Jupiter"]
        }
    ]

def main():
    # Initialize evaluator
    evaluator = PhiSQuADEvaluator()
    
    print("Phi Model SQuAD Evaluation")
    print("="*50)
    
    # Choose evaluation type
    eval_choice = input("Choose evaluation:\n1. SQuAD dataset subset\n2. Custom QA pairs\n3. Both\nEnter choice (1/2/3): ").strip()
    
    results = {}
    
    if eval_choice in ['1', '3']:
        # Evaluate on SQuAD subset
        num_samples = int(input("Enter number of SQuAD samples to evaluate (default 10): ") or "10")
        print(f"\nEvaluating on {num_samples} SQuAD samples...")
        squad_results = evaluator.evaluate_squad_subset(num_samples=num_samples)
        results['squad'] = squad_results
        
        print("\n" + "="*50)
        print("SQuAD EVALUATION RESULTS")
        print("="*50)
        print(f"Exact Match: {squad_results['exact_match']:.4f}")
        print(f"F1 Score: {squad_results['f1']:.4f}")
        print(f"Samples: {squad_results['num_samples']}")
    
    if eval_choice in ['2', '3']:
        # Evaluate on custom QA pairs
        print(f"\nEvaluating on custom QA pairs...")
        custom_qa = create_sample_qa_pairs()
        custom_results = evaluator.evaluate_custom_qa(custom_qa)
        results['custom'] = custom_results
        
        print("\n" + "="*50)
        print("CUSTOM QA EVALUATION RESULTS")
        print("="*50)
        print(f"Exact Match: {custom_results['exact_match']:.4f}")
        print(f"F1 Score: {custom_results['f1']:.4f}")
        print(f"Samples: {custom_results['num_samples']}")
    
    # Save results to file
    with open('phi_squad_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: phi_squad_evaluation_results.json")

if __name__ == "__main__":
    main()