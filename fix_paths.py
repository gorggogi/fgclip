import json

path = r'c:\Users\Charlene C. Dilig\OneDrive\Documents\Github\fgclip\fgclipEvaluation_V2_2_1200_HNM.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if '../captions/' in src or '../images' in src:
            new_src = src
            new_src = new_src.replace("'../captions/captions_datasetV2.csv'", "'captions/dataset_captionsV2.csv'")
            new_src = new_src.replace("'../images'", "'images'")
            if src != new_src:
                cell['source'] = [new_src]
                changes += 1
                idx = src.find('CSV_PATH')
                print('Fixed:', repr(src[idx:idx+80]))

print('Total cells updated:', changes)
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook saved OK.')
