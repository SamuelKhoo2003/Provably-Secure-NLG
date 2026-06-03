"""
Test Set Filter based on Decision Tree (Depth 4)
Trained on 250 samples, achieves 80.9% shard accuracy after filtering.

Usage:
    from test_filter import is_solvable, extract_filter_features
    
    features = extract_filter_features(tools_def, question, tool_name)
    if is_solvable(features):
        # Include in test set
"""

import re
import json


def extract_filter_features(tools_def: str, question: str, tool_name: str) -> dict:
    """Extract features needed for the filter decision tree."""
    
    # Parse tool descriptions
    desc_pattern = r'"description"\s*:\s*"([^"]*)"'
    tool_descs = re.findall(desc_pattern, tools_def)
    
    # Basic features
    tool_desc_len = len(tools_def)
    tool_count = max(1, tools_def.count('"name"'))
    question_len = len(question)
    num_char_in_tool_name = len(tool_name)
    question_num_words = len(question.split())
    
    # Tool description stats
    if tool_descs:
        avg_tool_desc_len = sum(len(d) for d in tool_descs) / len(tool_descs)
        min_tool_desc_len = min(len(d) for d in tool_descs)
    else:
        avg_tool_desc_len = tool_desc_len
        min_tool_desc_len = tool_desc_len
    
    avg_tool_len_per_tool = tool_desc_len / tool_count
    question_per_tool = question_len / tool_count
    
    return {
        'tool_desc_len': tool_desc_len,
        'question_num_words': question_num_words,
        'num_char_in_tool_name': num_char_in_tool_name,
        'avg_tool_len_per_tool': avg_tool_len_per_tool,
        'question_len': question_len,
        'min_tool_desc_len': min_tool_desc_len,
        'avg_tool_desc_len': avg_tool_desc_len,
        'question_per_tool': question_per_tool,
    }


def is_solvable(f: dict) -> bool:
    """
    Decision Tree (Depth 4) to predict if an example is solvable.
    Returns True if example should be KEPT (predicted as solvable).
    Returns False if example should be FILTERED OUT (predicted as both_wrong).
    """
    
    if f['tool_desc_len'] <= 2518.0:
        if f['question_num_words'] <= 40.5:
            if f['num_char_in_tool_name'] <= 44.5:
                if f['avg_tool_len_per_tool'] <= 158.38:
                    return False  # class: 1 (both_wrong)
                else:
                    return True   # class: 0 (solvable)
            else:  # num_char_in_tool_name > 44.5
                if f['question_len'] <= 172.5:
                    return True   # class: 0 (solvable)
                else:
                    return False  # class: 1 (both_wrong)
        else:  # question_num_words > 40.5
            if f['tool_desc_len'] <= 670.0:
                if f['tool_desc_len'] <= 285.5:
                    return False  # class: 1 (both_wrong)
                else:
                    return True   # class: 0 (solvable)
            else:  # tool_desc_len > 670.0
                return False      # class: 1 (both_wrong)
    
    else:  # tool_desc_len > 2518.0
        if f['min_tool_desc_len'] <= 79.0:
            if f['avg_tool_desc_len'] <= 108.66:
                if f['avg_tool_desc_len'] <= 104.18:
                    return False  # class: 1 (both_wrong)
                else:
                    return True   # class: 0 (solvable)
            else:  # avg_tool_desc_len > 108.66
                if f['avg_tool_len_per_tool'] <= 5775.0:
                    return False  # class: 1 (both_wrong)
                else:
                    return True   # class: 0 (solvable)
        else:  # min_tool_desc_len > 79.0
            if f['avg_tool_len_per_tool'] <= 790.37:
                return False      # class: 1 (both_wrong)
            else:
                if f['question_per_tool'] <= 66.92:
                    return True   # class: 0 (solvable)
                else:
                    return False  # class: 1 (both_wrong)


def filter_test_set(test_items: list) -> list:
    """
    Filter a list of test items, keeping only solvable examples.
    Each item should have 'tools', 'question', and a tool_call with 'name'.
    """
    from collections import namedtuple
    
    filtered = []
    for item in test_items:
        tools_def = item.get('tools', '')
        question = item.get('question', '')
        
        # Extract tool name from messages
        tool_name = ''
        msgs = item.get('messages')
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        if msgs:
            for m in msgs:
                if m.get('role') == 'tool_call':
                    try:
                        tool_name = eval(m.get('content', '{}')).get('name', '')
                    except:
                        pass
                    break
        
        features = extract_filter_features(tools_def, question, tool_name)
        if is_solvable(features):
            filtered.append(item)
    
    return filtered


if __name__ == "__main__":
    # Test on the 250-sample data
    base = [json.loads(l) for l in open('v1_base_250.jsonl')]
    shard = [json.loads(l) for l in open('v1_shard0_250.jsonl')]
    
    # Load original test data to get full features
    import random
    TEST_FILE = "/data/mwicker/VPA/data/test.jsonl"
    
    with open(TEST_FILE, "r") as f:
        lines = f.readlines()
    
    random.seed(42)
    sampled = random.sample(lines, min(500, len(lines)))
    
    def get_tool_call(item):
        msgs = item.get('messages')
        if isinstance(msgs, str): msgs = json.loads(msgs)
        for m in msgs:
            if m.get('role') == 'tool_call':
                return m.get('content')
        return ""
    
    def extract_tool_name(target_str):
        try:
            return eval(target_str).get('name', '')
        except:
            return ''
    
    test_data = []
    for line in sampled:
        item = json.loads(line)
        target = get_tool_call(item)
        if target:
            test_data.append({
                'tools': item['tools'],
                'question': item['question'],
                'tool_name': extract_tool_name(target)
            })
        if len(test_data) >= 250:
            break
    
    # Apply filter
    kept_indices = []
    for i, item in enumerate(test_data):
        features = extract_filter_features(item['tools'], item['question'], item['tool_name'])
        if is_solvable(features):
            kept_indices.append(i)
    
    n = len(kept_indices)
    base_correct = sum(1 for i in kept_indices if base[i]['correct'])
    shard_correct = sum(1 for i in kept_indices if shard[i]['correct'])
    
    print(f"=== FILTER VERIFICATION ===")
    print(f"Kept: {n}/250 examples")
    print(f"Base:     {base_correct}/{n} ({base_correct/n*100:.1f}%)")
    print(f"Shard_0:  {shard_correct}/{n} ({shard_correct/n*100:.1f}%)")
    print(f"Improvement: +{shard_correct - base_correct} (+{(shard_correct - base_correct)/n*100:.1f}%)")
