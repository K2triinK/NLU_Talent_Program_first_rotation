from bertopic import BERTopic
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from hdbscan import HDBSCAN
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
import yaml
from peft import PeftModel, PeftConfig
import torch

credentials = yaml.load(open('./credentials.yml'),  Loader=yaml.FullLoader)
AUTH_TOKEN = credentials['credentials']['auth_token']


def get_topic_headings(dict_topic_info, lang, adapter=False, instruct = True):
    
    # Load model
    device = "cuda:0"
    if instruct:
        model_name = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct"
    else:
        model_name = "AI-Sweden-Models/gpt-sw3-6.7b-v2"
    if adapter:
        peft_model_id = "./adapters/naming_topics_v3"
        #config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', use_auth_token=AUTH_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
        #print(config)
        #print(len(tokenizer))
        #print(model.config.vocab_size)
        # Load the Lora model
        model = PeftModel.from_pretrained(model, peft_model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
        model.to(device)
    
    # Prompting
    topic_headings = []
    prompt = "Create a short category heading that best describes what the keywords have in common.\n###\nKeywords: nuclear reactor, generator, solar energy, hydroelectric dam, coal, natural gas\nReasoning: All of the keywords are related to energy and more specifically its production.\nCategory: Energy Production\n###\nKeywords: car, airplane, banana, horse, apple, ship, motorbike, pear, bed, cupboard\nReasoning: This seems to be a mix of different things. The keywords include fruit but also forms of transport and items of furniture.\nCategory: Transportation, Fruit and Furniture\n###\nKeywords: covid, ebola, virus, vaccine, bacteria, antibiotics, immune system\nReasoning: The keywords have to do with diseases caused by viruses or bacteria.\nCategory: Virus and Bacteria\n###\n"
    prompt_sv = "Skapa en kort rubrik som bäst beskriver det som nyckelorden har gemensamt.\n###\nNyckelord: kärnreaktor, generator, solenergi, vattenkraftsdamm, kol, naturgas\nResonemang: Alla nyckelord är relaterade till energi och mer specifikt dess produktion.\nKategori: Energiproduktion\n###\nNyckelord: bil, flygplan , banan, häst, äpple, skepp, motorcykel, päron, säng, skåp\nResonemang: Det här verkar vara en blandning av olika saker. Nyckelorden inkluderar frukt men också transportformer och möbler.\nKategori: Transport, Frukt och Möbler\n###\nNyckelord: covid, ebola, virus, vaccin, bakterier, antibiotika, immunsystem\nResonemang: alla nyckelord har att göra med sjukdomar orsakade av virus eller bakterier.\nKategori: Virus och bakterier\n###\n"
    
    for k, v in dict_topic_info.items():
        keywords = [item[0] for item in v]
        keywords.reverse()
        keywords = ','.join(keywords)
        print(keywords)
        if adapter:
            encoded_sent = tokenizer(f"Keywords : {keywords} Topic Heading :", return_tensors='pt', truncation=True, max_length=2048)["input_ids"]
            gen_texts= []
            for _ in range(10):
                gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=50, temperature = 0.3, do_sample=True)[0]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).replace(f"Keywords : {keywords} Topic Heading : ", "")
                print(gen_text)
                gen_texts.append(gen_text)
        elif lang == 'EN':
            current_prompt = prompt
            if instruct:
                prompt_text = current_prompt + f"Keywords: {keywords}\nReasoning:"
                input_ids = [3, *tokenizer(f'User: {prompt_text}')['input_ids'], 2, *tokenizer('Bot:')['input_ids']]
                input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
                gen_texts = []
                for _ in range(10):
                    gen_tokens = model.generate(inputs = input_ids, max_new_tokens=400, temperature = 0.5, eos_token_id=[2,3])[0]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).split('Bot:')[1].strip()
                    gen_texts.append(gen_text.split('\n')[1].replace('Category:', '').strip())
            else:
                encoded_sent = tokenizer(current_prompt + f"Keywords: {keywords}\nReasoning:", return_tensors='pt', truncation=True, max_length=2048)["input_ids"]
                gen_texts = []
                for _ in range(10):
                    gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=50, temperature = 0.3, do_sample=True)[0]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).replace(current_prompt + f"Keywords: {keywords}\nReasoning:", "")
                    gen_texts.append(gen_text.split('\n')[1].replace('Category:', '').strip())
        else:
            current_prompt = prompt_sv
            if instruct:
                prompt_text = current_prompt + f"Nyckelord: {keywords}\nResonemang:"
                input_ids = [3, *tokenizer(f'User: {prompt_text}')['input_ids'], 2, *tokenizer('Bot:')['input_ids']]
                input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
                gen_texts = []
                for _ in range(10):
                    gen_tokens = model.generate(inputs = input_ids, max_new_tokens=400, temperature = 0.5, eos_token_id=[2,3])[0]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).split('Bot:')[1].strip()
                    gen_texts.append(gen_text.split('\n')[1].replace('Kategori:', '').strip())
            else:
                encoded_sent = tokenizer(current_prompt + f"Nyckelord: {keywords}\nResonemang:", return_tensors='pt', truncation=True, max_length=2048)["input_ids"]
                gen_texts = []
                for _ in range(10):
                    gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=50, temperature = 0.3, do_sample=True)[0]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).replace(current_prompt + f"Nyckelord: {keywords}\nResonemang:", "")
                    gen_texts.append(gen_text.split('\n')[1].replace('Kategori:', '').strip())
        print("generated texts: ", gen_texts)
        categories = sorted(gen_texts, key=gen_texts.count, reverse=True)
        words = [] #check for commas
        conj = 'och' if lang=='SV' else 'and'
        if adapter:
            conj = '&'
        for item in categories:
            if ',' in item:
                if lang == 'SV' and not adapter:
                    words += [w.replace(conj, '').strip() for w in item.split(',')]
            elif 'and' in item or '&' in item:
                words += item.split(f' {conj} ')
            elif item.replace(' ', '').isalpha():
                words.append(item)
        words = sorted(words, key=[w.lower() for w in words].count, reverse=True)
        heading = ''
        for item in words:
            if item.lower() not in heading.lower() and len(heading + ' ' + item) < 40:
                heading += ' ' + item + ',' 
        heading = heading[1:-1]
        heading_final = heading #delete this
        heading_final = ','.join(heading.split(',')[:-1]) + f' {conj} ' + heading.split(',')[-1].strip()
        if heading_final.strip().startswith(conj):
            heading_final = heading_final.strip()[len(conj):].strip()
        print("final heading: ", heading_final)
        topic_headings.append(heading_final)
    del model, tokenizer
    return topic_headings


def discover_topics(file_path, lang, num_topics, outliers, topk=10, measure='c_v'):
    # Read file
    df = pd.read_excel(file_path.name)
    
    # Load language model
    if lang == 'EN':
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        nlp = spacy.load("en_core_web_lg")
        language="english"
    elif lang == 'SV':
        #embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        embedding_model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')
        nlp = spacy.load("sv_core_news_lg")
        language="multilingual"
    nlp_docs = [item.replace("Denna text är maskinöversatt", "").replace("Direkt översatt från engelska", '') for item in list(df.text)]
    if 'title' in df.columns:
        titles = df.title.tolist()
    else:
        titles = None

    # Extract all pos-tags and lemmas from documents
    nlp_tags_list = []
    index = 0
    for i, doc in enumerate(nlp.pipe(nlp_docs)):
        for token in doc:
            temp = []
            temp.append(token.text)
            temp.append(token.lemma_)
            temp.append(token.pos_)
            temp.append(i)
            nlp_tags_list.append(temp)
        index += 1
   
    # Select tags and embed documents
    nlp_tags_df_fulltext = pd.DataFrame(nlp_tags_list, columns=['word', 'lemma', 'pos_tag', 'doc_id'])
    nlp_tags_topic_modelling = nlp_tags_df_fulltext[(nlp_tags_df_fulltext['pos_tag'].isin(['NOUN', 'VERB', 'PROPN', 'ADJ'])) & (nlp_tags_df_fulltext['lemma'].str.len() > 3)]
    titles = [titles[i] for i, doc in enumerate(nlp_docs) if i in set(nlp_tags_topic_modelling.doc_id)]
    project_ids = df.project_id.tolist()
    project_ids = [project_ids[i] for i, doc in enumerate(nlp_docs) if i in set(nlp_tags_topic_modelling.doc_id)]
    docs = [doc for i, doc in enumerate(nlp_docs) if i in set(nlp_tags_topic_modelling.doc_id)]
    embeddings = embedding_model.encode(docs)
    final_docs = list(nlp_tags_topic_modelling.groupby('doc_id').agg({'lemma' : lambda x: x.tolist()}).lemma)
    
    
    num_neighbors = min(50, int(len(final_docs)/4))
    num_components = min(100, len(final_docs)-2)
    
    # UMAP model
    umap_model = UMAP(n_neighbors=num_neighbors, n_components=20, min_dist=0.0, metric='cosine')
    
    # Cluster model
    if outliers == 'Disallow outliers':
        cluster_model = KMeans(n_clusters=2 * num_topics)
    else:
        min_cluster_size = min(10, int(len(final_docs)/10))
        cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    # Find most common labels
    allWordDist = nltk.FreqDist(w.lower() for w in list(nlp_tags_topic_modelling.lemma))
    mostCommon = allWordDist.most_common(int(0.005*len(allWordDist)))
    
    # Vectorizer model
    if len(final_docs) > 100:
        vectorizer_model = CountVectorizer(stop_words = [mostCommon[i][0] for i, item in enumerate(mostCommon)], min_df=0.01, max_df=0.5, ngram_range=(1,2))
    else:
        vectorizer_model = CountVectorizer(stop_words = [mostCommon[i][0] for i, item in enumerate(mostCommon)], max_df=0.5, ngram_range=(1,2))
    ctfidf_model = ClassTfidfTransformer()
    minimum_topic_size = 10 if len(final_docs) > 50 else 5
    
    # Perform topic modeling
    topic_model = BERTopic(umap_model = umap_model,
                hdbscan_model = cluster_model,
                vectorizer_model = vectorizer_model,
                ctfidf_model = ctfidf_model,
                min_topic_size = minimum_topic_size,
                nr_topics = num_topics,
                language=language) #,seed_topic_list=seed_topic_list
    
    topics, _ = topic_model.fit_transform([' '.join(item) for item in final_docs], embeddings)
    
    #Score the topic model
    #Coherence
    num_topics = len(set([t for t in topics if t != -1]))
    bertopic_topics = []
    for i in range(num_topics):
        temp = []
        for item in topic_model.get_topic(i):
            if not ' ' in item:
                if item[0] in nlp_tags_topic_modelling.lemma.tolist():
                    temp.append(item[0])
            else:
                w1 = item[0].split()[0]
                w2 = item[0].split()[1]
                if w1 in nlp_tags_topic_modelling.lemma.tolist():
                    temp.append(w1)
                if w2 in nlp_tags_topic_modelling.lemma.tolist():
                    temp.append(w2)
        bertopic_topics.append(temp[:topk])
    cv = CoherenceModel(topics=bertopic_topics, texts = final_docs, 
                        dictionary=Dictionary(final_docs), coherence=measure, processes=1, topn=topk)
    coherence_score = cv.get_coherence()
    
    #Diversity
    unique_words = set()
    for topic in bertopic_topics:
        unique_words = unique_words.union(set(topic[:topk]))
    diversity_score = len(unique_words) / (topk * len(bertopic_topics))
    
    topic_info = topic_model.get_topics()
    print(topic_info)
    
    # Give labels to the topics
    topic_labels = get_topic_headings(topic_info, lang)
        
    if -1 not in topic_info:    
        topic_model.set_topic_labels([f"{item} ({topics.count(i)})" for i, item in enumerate(topic_labels)])
    else:
        topic_model.set_topic_labels([f"{item} ({topics.count(i)})" for i, item in zip(list(range(-1, len(topic_labels))), topic_labels)])
    
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    visualize_texts = docs if not titles else titles
    
    # Visual representation of the modeling
    plot = topic_model.visualize_documents(visualize_texts, reduced_embeddings=reduced_embeddings, hide_document_hover=False,
                                          custom_labels = True)
    #plot.write_html("result_sheets/plot_html.html")
    results_table = topic_model.get_document_info(docs)
    clusters = results_table[['Topic', 'Name', 'CustomName']]
    
    df_projects = pd.DataFrame()
    df_projects['project_id'] = project_ids
    df_projects['title'] = titles
    df_projects['text'] = final_docs
    df_projects['cluster'] = clusters['Topic'].tolist()
    df_projects['cluster_keywords'] = clusters['Name'].tolist()
    df_projects['cluster_name'] = clusters['CustomName'].tolist()
    df_final = df.merge(df_projects[['project_id', 'cluster', 'cluster_keywords', 'cluster_name']], on='project_id', how='left')
    df_final.to_excel("result_sheets/discover_results.xlsx", index = False)   
    return plot, round(100 * coherence_score), round(100 * diversity_score), df_final, "result_sheets/discover_results.xlsx"