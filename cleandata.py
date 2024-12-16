import chardet, json

# initialize the trec06c_clean.csv file
with open('trec06c_clean.csv', 'w') as f:
    f.write('Category,Message\n')
# initialize the trec06c_clean.json file
with open('trec06c_clean.json', 'w') as f:
    f.write('')

path = '/data/jiani/prompt/Bert_tuning_for_jailbreak/spam_email_detect/trec06c/full/index'

def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

lines = read_file(path)

for line in lines:
    line = line.strip().split(' ')
    category = line[0]
    text_path = line[1]
    content = ''
    with open(text_path, 'rb') as f:
        raw_data = f.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        if detected_encoding is None:
            continue
        print("Detected encoding:", detected_encoding)
    with open(text_path, 'r', encoding=detected_encoding, errors='ignore') as f:
        text = f.read()
        # first time detect \n\n and take the second part
        text = text.split('\n\n', 1)[1]
        content = text
    # save to new jsonl file
    with open('trec06c_clean.json', 'a') as f:
        f.write(json.dumps({'Category': category, 'Message': content}, ensure_ascii=False) + '\n')
        
# make a small test set
with open('trec06c_clean.json', 'r') as f:
    lines = f.readlines()
    with open('trec06c_test.json', 'w') as f:
        for i in range(10000):
            f.write(lines[i])
       


