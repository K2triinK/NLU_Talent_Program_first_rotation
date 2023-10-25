from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import spacy
import regex
from Levenshtein import distance as lev
import yaml
from peft import PeftModel, PeftConfig

nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
credentials = yaml.load(open('./credentials.yml'),  Loader=yaml.FullLoader)
AUTH_TOKEN = credentials['credentials']['auth_token']

def lemmatize(keyword, tokens, lemmas):
    # Obtain lemmas of keywords
    if '-' in keyword:
        return keyword
    try:
        k = ''
        for i, w in enumerate(keyword.split()):
            if w in lemmas:
                k += w + ' '
            elif w in tokens:
                k += lemmas[tokens.index(w)] + ' '
    except:
        k = keyword
    return k.strip()


def clean_keywords(text, project_desc, n=10):
    if text.startswith("This is a keyword extraction model"):
        text = text.split('Keywords:')[-1]
    if text.startswith(" "):
        keywords = text.split('\n')[0].strip().lower()
        keywords = [re.sub('[0-9]+$', ' ', k).strip() for k in keywords.split(',')[:2*n]]
        keywords_temp = []
        for k in keywords:
            parts = k.split(' and ')
            keywords_temp += parts
        keywords = keywords_temp
        normalized_keywords = []
        ignored_keywords = []
        doc = nlp(project_desc.lower())
        token_texts = []
        token_lemmas = []
        for token in doc:
            token_texts.append(token.text)
            token_lemmas.append(token.lemma_)
        for item in keywords:
            item = item.split('(')[0].strip()
            if ' ' + item in project_desc.lower() or item + ' ' in project_desc.lower():
                if ' ' not in item and item in token_lemmas:
                    normalized_keywords.append(item)
                elif ' ' not in item and item in token_texts:
                    normalized_keywords.append(token_lemmas[token_texts.index(item)])
                else:
                    normalized_keywords.append(item)
            elif '/' in item:
                words = item.split('/')
                for w in words:
                    if w in token_lemmas:
                        normalized_keywords.append(w)
                    elif w in token_texts:
                        normalized_keywords.append(token_lemmas[token_texts.index(w)])
            elif item in token_lemmas:
                normalized_keywords.append(item)
            elif item in token_texts:
                normalized_keywords.append(token_lemmas[token_texts.index(item)])
            elif item.endswith(')') and item.split('(')[0] in project_desc.lower():
                normalized_keywords.append(item.split('(')[0])
            elif ("'" in item and (item[:item.index("'")] in project_desc.lower() and 
                                                item[item.index("'")+1:] in project_desc.lower())):
                normalized_keywords.append(item)
            elif item.endswith('ion') and item[:-3] + 'ing' in project_desc.lower():
                normalized_keywords.append(item)
            elif item.endswith('ion') and item[:-3] + 'ive' in project_desc.lower():
                normalized_keywords.append(item)
            elif item.endswith('ity') and (item[:-3] in token_texts or item[:-3] in token_lemmas):
                normalized_keywords.append(item)
            elif item.endswith('s') and (item[:-1] in token_texts or item[:-1] in token_lemmas):
                normalized_keywords.append(item[:-1])
            elif item.endswith('y') and (item[:-1] in token_texts or item[:-1] in token_lemmas):
                normalized_keywords.append(item)
            elif item.endswith('ies') and (item[:-3] + 'y' in token_texts or item[:-3] + 'y' in token_lemmas):
                normalized_keywords.append(item[:-3] + 'y')
            elif '-' in item and item.replace('-', '') in project_desc.lower():
                normalized_keywords.append(item.replace('-', ''))
            elif '-' in item and item.replace('-', ' ') in project_desc.lower():
                normalized_keywords.append(item.replace('-', ' '))
            elif ' ' in item or '-' in item:
                item = item.replace('-', ' ')
                words = item.split()
                add = True
                for w in words:
                    if w not in token_texts and w not in token_lemmas:
                        add = False
                        break
                if add:
                    normalized_keywords.append(item)
                elif len(words) > 2 and (f"{words[0]} {words[1]}" in project_desc.lower() or f"{words[0]}-{words[1]}" in project_desc.lower()):
                    normalized_keywords.append(f"{words[0]} {words[1]}")
                elif len(words) > 2 and (f"{words[1]} {words[2]}" in project_desc.lower() or f"{words[1]}-{words[2]}" in project_desc.lower()):
                    normalized_keywords.append(f"{words[1]} {words[2]}")
                else:
                    ignored_keywords.append(item)
            else:
                ignored_keywords.append(item)
                
        #Double check ignored keywords
        for ignored in ignored_keywords:
            if '(' not in ignored and ')' not in ignored:
                pattern = f"{ignored[:2]}[a-y-]*\s?[a-y-]*{ignored[-2:]}"
                query_string = project_desc.lower()
                candidates = [item.strip() for item in regex.findall(pattern, query_string)]
                for c in set(candidates):
                    if lev(c, ignored) == 1 and ignored not in normalized_keywords:
                        try:
                            normalized_keywords.append(c)
                            ignored_keywords.remove(ignored)
                        except:
                            pass
        normalized_keywords = [lemmatize(k, token_texts, token_lemmas) for k in normalized_keywords if len(k) > 2]
        final_keywords = []
        used_words = []
        for i, key in enumerate(normalized_keywords[:-1]):
            if key in used_words:
                continue
            if ' ' not in key and f"{key} {normalized_keywords[i+1]}" in project_desc.lower():
                final_keywords.append(f"{key} {normalized_keywords[i+1]}".strip())
                used_words.append(key)
                used_words.append(normalized_keywords[i+1])
            else:
                final_keywords.append(key.strip())
        if normalized_keywords and normalized_keywords[-1] not in used_words:
            final_keywords.append(normalized_keywords[-1].strip())
        final_keywords = [k for k in final_keywords if len(k) > 2]
        return (final_keywords[:n], ','.join(ignored_keywords))
    return ([], '')


def clean_keywords_adapter(prediction, input_text):
    chosen_keywords = []
    ignored_keywords = []
    predicted_keywords = prediction.split(', ')
    predicted_keywords[0] = predicted_keywords[0].lower()
    predicted_keywords = [item.replace('\n', '').strip() for item in predicted_keywords][:-1]
    for w in predicted_keywords:
        if w in input_text:
            if w not in chosen_keywords:
                chosen_keywords.append(w)
        else:
            ignored_keywords.append(w)
    return chosen_keywords[:10], ignored_keywords


def get_keywords(input_text, model=None, tokenizer=None, device=None, adapter=False, instruct=False):
    model_existed = True
    if not model and not tokenizer and not device:
        model_existed = False
        device = "cuda:0"
        if adapter:
            model_name = "AI-Sweden/gpt-sw3-6.7b-private"
            peft_model_id = "./adapters/extract_keywords_v1"
            config = PeftConfig.from_pretrained(peft_model_id)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto', use_auth_token=AUTH_TOKEN)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=AUTH_TOKEN)
            # Load the Lora model
            model = PeftModel.from_pretrained(model, peft_model_id)
            text = f"Project text : {input_text} Keywords : "
            encoded_sent = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048-50)["input_ids"]
            gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=50, temperature=0)[0]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            print(gen_text)
            prediction_text = gen_text.split("Keywords :")[1]
            print(prediction_text)
            chosen_keywords, ignored_keywords = clean_keywords_adapter(prediction_text, input_text)
            print('Chosen:', chosen_keywords)
            print('Ignored:', ignored_keywords)
            return chosen_keywords #prediction_text.split(',')
        elif instruct:
            model_name = "AI-Sweden/gpt-sw3-6.7b-v2-instruct-private" #"AI-Sweden/gpt-sw3-6.7b-v2-instruct-no-dolly-private"
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
            model.to(device)
            prompt = f"Extract ten most significant keywords from the text.\nProject text: {input_text}\nKeywords:"
            input_ids = [3, *tokenizer(f'User: {prompt}')['input_ids'], 2, *tokenizer('Bot:')['input_ids']]
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
            gen_tokens = model.generate(inputs = input_ids, max_new_tokens=50, temperature = 0, eos_token_id=[2,3])[0]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).split('Bot:')[1].strip()
            print(gen_text)
            return gen_text.split(',')
        else:
            model_name =  "AI-Sweden/gpt-sw3-6.7b-v2-private" #"togethercomputer/GPT-JT-6B-v1"
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
            model.to(device)
        
    examples = "This is a keyword extraction model whose task is to output ten keywords for every document it sees.\n###\nInstitute of Animal Reproduction and Food Research of the Polish Academy of Sciences (IAR&FR), in partnership with Olsztyn School of Higher Education (OSHE) invites kids, tweens, teens and adults to join FUSION Researchers’ Night – a unique occasion when science, entertainment and education meet in one place! Fusion Night is an incredible blend of research, innovation, fun and education that introduces young people to meet scientists who live in their city but whose work reaches across the whole world. The FUSION Night (FN) programmes revolves around the activities that will increase the awareness of the community at large about the importance of research careers and the central role of scientists in improving the daily life and well-being of European citizens. Visitors of the FN will have a chance to join edutaining activities, become a researcher for one night and give lecture to their peers (Polish Academy of Kids), test the content of vitamins in juice or milk (“Vitamins fusion or health illusion?), discover the secrets of cryopreservation in a mobile laboratory (“Ice land”), develop their own creams using fruit powders (“Tasty skin”), design a book from scratch (“Night in the Library”), learn the secrets of animal biodiversity (“Take a walk on the wide side”) or make a fruit puree in liquid nitrogen (“Molecular Show of Fusion Kitchen”). Sport enthusiasts can expect swimming competitions, basketball game, salsa course and martial arts training all conducted by our young researchers. We haven’t forgotten about the music and art lovers too, preparing for them a live concert of an international group of English teachers and an exhibition of science curiosities captured on the stained glass. Collaboration with partners from industrial circles will help to provide visitors with a first-hand view of the products and innovations “born” thanks to the solutions provided by science and researchers.\nKeywords: research, science, entertainment, education, activity, awareness, community, visitor, solution, importance\n###\nCurrently freight transport represents 40% of the total transport emission and 32% in urban area. Many initiatives are under development to reduce costs and negative impact of freight and service trip in urban area. Some of them concern supply chain improvements and more specifically consolidation centre projects. Few study cases are dedicated to construction industry. However, urban population tends to grow, increasing the need to develop and reconstruct urban centres. Construction material logistic impact in urban area will intensify in terms of costs and negative impacts in urban area. Yet, only few experiences of Construction Consolidation Centres can be found.Among these initiatives, four are construction site specific (Stockholm, Utrecht, Berlin, London Heathrow) and only one is dedicated to several construction projects (London CC). Theses pilots studies have demonstrated reduced transportation impacts, positive effects on transportation efficiency and construction site productivity.Several limitations to the transferability of this concept are identified: one on hand the demonstrators were implemented in specific contexts (regulatory incentives, cities investment contribution, and specifics transport and logistics infrastructure issues) which are not the same in France, Spain, Italy and Luxembourg. On the other hand, economic viability has not been demonstrated.The project addresses the different requirements for transferability of supply chain optimization concepts as well as CCCs and new ways of working between supply chain stakeholders. The approach is to identify an integrated collaborative approach and business model among construction supply chain actors. Three main steps will be performed: analyse the current issues along the construction supply chain, propose several optimization scenarios regarding these issues, simulate and analyse costs optimization and environmental impacts to propose new partnership opportunities based on savings distribution\nKeywords: freight, transport, emission, urban, supply chain, construction, optimization, cost, environment, partnership\n###\n"


    sents = sent_tokenize(input_text)
    t = []
    for sent in sents:
        if len(' '.join(t + [sent])) < 2100:
            t.append(sent)
        else:
            break
    t = ' '.join(t)
    text = f"{examples}{t}\nKeywords:"
    encoded_sent = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048-50)["input_ids"]
    if 'AI-Sweden' in model_name:
        gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=50, temperature=0)[0]
    else:
        gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=50, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0)[0]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    prediction_text = gen_text.replace(text, '')
    #print(prediction_text)
    torch.cuda.empty_cache()
    chosen_keywords, ignored_keywords = clean_keywords(prediction_text, input_text)
    if not model_existed:
        del model, tokenizer
    return chosen_keywords #, ignored_keywords