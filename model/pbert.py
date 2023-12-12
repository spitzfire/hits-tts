from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence
import torch

# Load XPhoneBERT model and its tokenizer
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")

# Load Text2PhonemeSequence
text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=False)

# Input sequence that is already WORD-SEGMENTED (and text-normalized if applicable)
sentence = "Unlike some , I know that this is a testing text ." 
sentence2 = "Unlike some , know that this is a testing text ."  


input_phoneme1 = text2phone_model.infer_sentence(sentence)
input_phoneme2 = text2phone_model.infer_sentence(sentence2)

input_ids = tokenizer(input_phoneme1, return_tensors="pt", padding=True)
print((input_ids["input_ids"].shape))


input_ids = tokenizer([input_phoneme1, input_phoneme2], return_tensors="pt", padding=True)
print(max(len(x) for x in input_ids["input_ids"]))
print((input_ids["input_ids"].shape))

# # print(input_phonemes)
# print((input_ids["input_ids"].numpy()))

# pad_token = tokenizer.pad_token

# # If you want to find the numerical ID of the pad token
# pad_token_id = tokenizer.pad_token_id

# # Print the pad token and its ID
# print("Pad Token:", pad_token)
# print("Pad Token ID:", pad_token_id)

with torch.no_grad():
    features = xphonebert(**input_ids)

print(features["pooler_output"].detach().numpy().shape)