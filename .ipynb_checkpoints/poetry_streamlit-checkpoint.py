import prosodic as p
import pronouncing
from wordfreq import word_frequency
import os
import streamlit as st
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM
import re

###############################################
#           RHYME AND METER PART              #
###############################################

def get_list_of_rhymes(last_word, top_n = 20):
    """
    Returns a list of rhymes for a given word ranked by their usage frequency
    """
    # get rhymes
    rhymes = pronouncing.rhymes(last_word)
    # remove words with punctuation
    rhymes = [x for x in rhymes if re.findall(r'[.?\-", ]+', x) == [] ]
    rhymes = list(set(rhymes))
    # sort them by usage frequency
    rhymes_sorted = sorted(rhymes, key = lambda x: word_frequency(x, 'en'), reverse = True)
    # only return top_n rhymes
    rhymes_sorted = rhymes_sorted[:top_n]
    return rhymes_sorted

def get_meter(line):
    """
    Returns the rhythmic structure of the line in a "swwsww" format,
    where "s" stands for stressed syllables and "w" stands for non-stressed syllables
    """ 
    text = p.Text(line)
    text.parse(meter = 'default_english')
    best_parses = text.bestParses()
    meter = best_parses[0].str_meter()
    return(meter)

def get_meter_string(line):
    """
    Returns the line with stressed syllables capitalized
    """ 
    text = p.Text(line)
    text.parse(meter = 'default_english')
    best_parses = text.bestParses()
    meter_string = best_parses[0]
    return(meter_string)

def get_last_word(line):
    #remove punctuation
    res = re.sub(r'[^\w\s]', '', line)
    # get the last word
    last_word = res.split(' ')[-1]
    return last_word

###############################################
#                 BERT PART                   #
###############################################
#https://github.com/renatoviolin/next_word_prediction/blob/master/main.py

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[0]:
        text_sentence += ' .'
        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def get_all_predictions(text_sentence, top_clean=5):
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    scores = predict[0, mask_idx, :].topk(100).values.tolist()
    values = decode(bert_tokenizer, predict[0, mask_idx, :].topk(100).indices.tolist(), top_clean).split('\n')
    #return {'values': values.split('\n'), 'scores': scores}
    return values

def get_prediction_eos(input_text):
    try:
        input_text = '<mask> ' + input_text
        res = get_all_predictions(input_text, top_clean=int(100))
        return res
    except Exception as error:
        print(error)
        pass
    
###############################################
#              FINDING MATCHES                #
###############################################    
    
def get_match_list(line_options, meter, top_n = 5):
    """
    This function takes a list of line ending options,
    finds top N previous words for each of them 
    and only returns new suggested endings
    that match the given meter
    """
    match_list = []
    try:
        for current_line in line_options:
            # predict some previous words
            suggest_list = get_prediction_eos(current_line)
            # exclude exceptions
            suggest_list = [x for x in suggest_list if x != '...']
            # only use top n suggestions
            suggest_list = suggest_list[:top_n]
            for suggest in suggest_list:
                # add the suggestion to the current line
                new_line = suggest + ' ' + current_line
                # print(new_line)
                # find the rhythmic structure of the new line
                current_meter = get_meter(new_line)
                # see if it matches the original meter
                match = meter[-len(current_meter):] == current_meter
                # if it does, add the new line to the match list
                if match:
                    match_list.append(new_line)
        return match_list
    except Exception:
        return match_list

def get_full_matches(line_options, meter, top_n = 5, n_stop = 10):
    full_matches = []
    # start with initial options
    matches_to_add = line_options
    # check that some matches were found and the full match limit isn't reached
    while matches_to_add != [] and len(full_matches) <= n_stop:
        #print(matches_to_add)
        # update the options list with the next word
        matches_to_add = get_match_list(matches_to_add, meter, top_n)
        # check if there are any full meter matches in the new list
        new_full_matches =  [x for x in matches_to_add if get_meter(x) == meter]
        # add new full matches to the output
        if new_full_matches != []:
            full_matches = full_matches + new_full_matches
            #print('FULL_MATCHES')
            #print(full_matches)
    return full_matches[:n_stop]

def get_partial_matches(line_options, meter, top_n = 5, n_stop = 20, n_words = 2):
    # TODO: Balance multiple rhymes
    partial_matches = []
    # start with initial options
    matches_to_add = line_options
    counter = 0
    # check that some matches were found and the full match limit isn't reached
    while matches_to_add != [] and counter < n_words:
        # update the options list with the next word
        matches_to_add = get_match_list(matches_to_add, meter, top_n)
        counter+=1
    # balance for different rhymes
    final_list = []
    if line_options != 0:
        n_each = round(n_stop/len(line_options))
    else: 
        n_each = 0
    for option in line_options:
        match_list = [x for x in matches_to_add if x.endswith(option)]
        final_list = final_list + match_list
    return final_list
#    return matches_to_add[:n_stop]

    
st.title("Poet's assistant")

#p.config['print_to_screen']=0 

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

line1 = st.text_input('First line', 'Nice to meet you, where you been?')

st.write(get_meter_string(line1))

last_word_1 = get_last_word(line1)

# TODO: add n_rhymes slider

rhymes_1 = get_list_of_rhymes(last_word_1, top_n = 20)

meter_1 = get_meter(line1)

options_1 = st.multiselect(
     'Choose some rhymes',
     rhymes_1,
     [])

n_words = st.slider('Suggested words', 0, 10, 0)

matches_1 = get_partial_matches(line_options=options_1, meter= meter_1, top_n = 5, n_stop = 20, n_words = n_words)
st.write(', '.join(matches_1))
    
if st.button('Get full suggestions'):
    full_matches_1 = get_full_matches(line_options = options_1, meter = meter_1, top_n = 3, n_stop = 10)
    st.write(', '.join(full_matches_1))
    
    


    
     