import json

path = r'c:\Users\Charlene C. Dilig\OneDrive\Documents\Github\fgclip\fgclipEvaluation_V2_2_1200_HNM.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'IMG_DIR' in src and ('images' in src or '../images' in src):
            idx = src.find('IMG_DIR')
            print('IMG_DIR context:', repr(src[idx:idx+80]))
            print('---')
