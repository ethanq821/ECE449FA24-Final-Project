import json

spam = 0
ham = 0
with open('/data/jiani/prompt/Bert_tuning_for_jailbreak/spam_email_detect/trec06c/codes/trec06c_clean.json') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data['Category'] == 'spam':
            spam += 1
        else:
            ham += 1

print("spam: ", spam)
print("ham: ", ham)