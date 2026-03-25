import pandas as pd
import json
from collections import Counter

# Load a sample to be efficient
df = pd.read_csv('data/raw/events.csv', nrows=50000)

qualifier_counts = Counter()

def extract_qualifiers(q_str):
    if not isinstance(q_str, str) or q_str == '[]':
        return
    try:
        q_list = json.loads(q_str.replace("'", '"'))
        for q in q_list:
            if isinstance(q, dict) and 'type' in q:
                if isinstance(q['type'], dict) and 'displayName' in q['type']:
                    qualifier_counts[q['type']['displayName']] += 1
                elif isinstance(q['type'], str):
                    qualifier_counts[q['type']] += 1
    except Exception as e:
        pass

df['qualifiers'].apply(extract_qualifiers)

print(f"Unique Qualifiers found in sample: {len(qualifier_counts)}")
print("Top 10 Qualifiers:", qualifier_counts.most_common(10))
