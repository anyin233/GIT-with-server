from dee.utils import BERTChineseCharacterTokenizer
BERT_MODEL = 'bert-base-chinese'

tokenizer = BERTChineseCharacterTokenizer.from_pretrained(BERT_MODEL)
tokens = [[(3330, 1290, 7471), (122, 123, 122, 126, 122, 121, 121, 121, 5500), (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385), (123, 123, 127, 122, 130, 130, 130, 130, 5500), (127, 119, 125, 122, 110), (122, 129, 123, 121, 121, 121, 121, 121, 5500), (123, 121, 122, 128, 2399, 122, 123, 3299, 128, 3189), None, None], [(3330, 1290, 7471), (122, 122, 129, 129, 127, 121, 121, 5500), (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385), (123, 123, 127, 122, 130, 130, 130, 130, 5500), (127, 119, 125, 122, 110), (122, 129, 123, 121, 121, 121, 121, 121, 5500), (123, 121, 122, 129, 2399, 130, 3299, 127, 3189), None, None]]
for token in tokens:
    for t in token:
        if not t is None:
            print("".join(tokenizer.convert_ids_to_tokens(t)))
        else:
            print("null")
    print('=' * 20)
