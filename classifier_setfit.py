from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import pandas as pd
import numpy as np
import math
import copy
from small_text import TextDataset
from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text.integrations.transformers.classifiers.factories import SetFitClassificationFactory
from small_text import (
    PoolBasedActiveLearner, 
    random_initialization_balanced,
    BreakingTies,
    SubsamplingQueryStrategy,
    EmptyPoolException
)
import gc
import torch
from sklearn.metrics import accuracy_score
from small_text.base import LABEL_UNLABELED
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from get_keywords import *
from transformers import BartForConditionalGeneration, BartTokenizer

credentials = yaml.load(open('./credentials.yml'),  Loader=yaml.FullLoader)
AUTH_TOKEN = credentials['credentials']['auth_token']
MODEL_NAME_ENG = "all-mpnet-base-v2"
MODEL_NAME_SV = "KBLab/sentence-bert-swedish-cased" #"paraphrase-multilingual-mpnet-base-v2"


########################################################################
# This is the code for augmenting data and therafter scoring the augmented examples 
# (scoring code from https://github.com/zhaominyiz/EPiDA/blob/main/eda.py)
########################################################################
EPS = 1e-10

def JointH(a,b):
    s = 0.0
    for i in range(a.size()[0]):
        for j in range(b.size()[0]):
            _a = a[i]
            _b = b[j]
            p = _a * _b
            if p<=1e-10:
                continue
            s += p*torch.log2(p)
    return s*-1.0

def mutal_info(a,b):
    return H(a)+H(b)-JointH(a,b)


def MI(z,zt):
    C = z.size()[1]
    # actually they are not independent
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    P = ((P + P.t())/2) / P.sum()
    P[(P<EPS).data] = EPS
    Pi = P.sum(dim=1).view(C,1).expand(C,C)
    Pj = P.sum(dim=0).view(1,C).expand(C,C)
    # revise by 1.0
    return 1.0-(P * (-torch.log(Pi)-torch.log(Pj)+torch.log(P))).sum()

def CEM(z,zt):
    return MI(z,zt)-H(z)

def H(P):
    P[(P<EPS).data] = EPS
    return -(P*torch.log(P)).sum()

def REM(z,zt):
    zt[(zt<EPS).data] = EPS
    return -torch.sum(z*torch.log(zt))

def gradmutualgain(label,one_hot,softmaxed,softmaxed_y,loss_fn=None):
    up = REM(softmaxed.unsqueeze(0),one_hot.unsqueeze(0))
    # make all the less than zero > 0
    down = 1.0+CEM(softmaxed.unsqueeze(0),softmaxed_y.unsqueeze(0))
    return up,down


def sort_augmented(txt,label,augmented_texts,model, num_aug=4):
    alpha_epda=0.5
    txts = list(set(augmented_texts))
    oldtxts = txts
    labels = []
    for i in range(len(txts)):
        labels.append(label)
    labels = torch.tensor(labels).long()
    ups,downs,scores = [],[],[]
    
    for i in range(len(oldtxts)):
        #softmaxed = torch.softmax(outputs[i],0)
        softmaxed = model.predict_proba([oldtxts[i]])[0]
        #print(softmaxed)
        #softmaxed = torch.from_numpy(softmaxed)
        c = model.predict_proba([txt])[0]
        #c = torch.from_numpy(c)
        C = softmaxed.size(0)
        one_hot = torch.zeros(C)
        one_hot[label] = 1.0
        _up,_down = gradmutualgain(label,one_hot,softmaxed,c,CosineSimilarityLoss)
        ups.append(_up)
        downs.append(_down)
    ups = np.array(ups)
    downs = np.array(downs)
    ups = (ups-np.min(ups))/(np.max(ups)-np.min(ups))
    downs = (downs-np.min(downs))/(np.max(downs)-np.min(downs))
    for i in range(downs.shape[0]):
        _up,_down=ups[i],downs[i]
        score = alpha_epda*_up + (1.0-alpha_epda)*_down
        if score == np.nan or math.isnan(score):
            score = 1.0
        scores.append(score)
    scores = np.array(scores)
    sortargs = np.argsort(-scores).tolist()
    newtxts = []
    newscores = []
    for idx in sortargs[:num_aug]:
        newtxts.append(oldtxts[idx])
        newscores.append(scores[idx])
    return newtxts,newscores


def get_augmented_data(examples, labels, classifier, examples_to_generate=4, lang='en', adapter = True):
    augmented_texts  = []
    augmented_labels = []
    device = "cuda:0"
    if adapter:
        model_name = "AI-Sweden/gpt-sw3-6.7b-private"
        peft_model_id = "./adapters/paraphrase_v1"
        config = PeftConfig.from_pretrained(peft_model_id)
        model_orig = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto', use_auth_token=AUTH_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=AUTH_TOKEN)
        model = PeftModel.from_pretrained(model_orig, peft_model_id)
    else:
        model_name = "AI-Sweden/gpt-sw3-6.7b-v2-instruct-private"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_name,use_auth_token=AUTH_TOKEN)
        model.to(device)
    if lang == 'en':
        model_emb = SentenceTransformer(MODEL_NAME_ENG)
    else:
        model_emb = SentenceTransformer(MODEL_NAME_SV)
    prompt_en = """Paraphrase the text so that it differs from the original text but has the same meaning.\n###\n[Original]: Algeria recalled its ambassador to Paris on Saturday and closed its airspace to French military planes a day later after the French president made comments about the northern Africa country.\n[Paraphrase]: Last Saturday, the Algerian government recalled its ambassador and stopped accepting French military airplanes in its airspace. It happened one day after the French president made comments about Algeria.\n###\n[Original]: A roadmap to the realization of fusion energy was adopted by the EFDA system at the end of 2012. The roadmap aims at achieving all the necessary know-how to start the construction of a demonstration power plant (DEMO) by 2030, in order to reach the goal of fusion electricity in the grid by 2050.\n[Paraphrase]: In 2012 the EFDA system adopted a roadmap to the realization of fusion energy. The goal is to achieve sufficient know-how to be able to start constructing a demo power plant by 2030. This is a step on the way to include fusion energy in the grid by 2050.\n###\n[Original]: The consortium will provide 3D-printers with high throughput and outstanding materials and energy efficiency. The project is clearly industrially driven, with 8 out of 10 partner being SMEs or industry. Targeted end-use applications include ceramics for aerospace engineering, medical devices and energy efficient lighting 
applications.\n[Paraphrase]: Great quality 3D-printers will be provided by the consortium. 80% of the project partners are SMEs or industry. Target applications include but are not limited to aerospace engineering and medical devices.\n###\n[Original]:"""
    prompt_sv = """Skriv om texten så att den skiljer sig från originaltexten men har samma innebörd.\n###\n[Original]: Algeriet återkallade sin ambassadör i Paris på lördagen och stängde sitt luftrum för franska militärplan en dag senare efter det franska president gjorde kommentarer om landet i norra Afrika.\n[Omskrivning]: I lördags återkallade den algeriska regeringen sin ambassadör och slutade acceptera franska militärflygplan i dess luftrum. Det hände en dag efter att den franske presidenten kommenterat Algeriet.\n###\n[Original]: En plan för förverkligandet av fusionsenergi antogs av EFDA-systemet i slutet av 2012. Kartan syftar till att uppnå all nödvändig kunskap för att påbörja byggandet av ett demonstrationskraftverk (DEMO) senast 2030, för att nå målet om fusionsel i elnätet till 2050.\n[Omskrivning]: 2012 antog EFDA-systemet en färdplan för att förverkliga fusionsenergi. Målet är att uppnå tillräckligt med kunnande för att kunna börja bygga ett demokraftverk till 2030. Detta är ett steg på vägen att till 2050 inkludera fusionsenergi i nätet.\n###\n[Original]: Konsortiet kommer att förse 3D-skrivare med hög genomströmning och enastående material och energieffektivitet. Projektet är tydligt industridrivet, där 8 av 10 partner är små och medelstora företag eller industri. Riktade slutanvändningstillämpningar inkluderar keramik för flygteknik, medicinsk utrustning och energieffektiv belysning.\n[Omskrivning]: 3D-skrivare av hög kvalitet kommer att tillhandahållas av konsortiet. 80 % av projektpartnerna är små och medelstora företag eller industri. Målapplikationer inkluderar men är inte begränsade till flygteknik och medicinsk utrustning.\n###\n[Original]:"""
    
    for ex, lab in zip(examples, labels):
        generated_texts = []
        for _ in range(5 * examples_to_generate):
            if adapter:
                adapter_prompt = f"Paraphrase the following text.\nProject text : {ex[:1600]}\nParaphrased text :"
                encoded_sent = tokenizer(adapter_prompt, return_tensors='pt', truncation=True, max_length=2048-400)["input_ids"]
                gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=400, temperature=0)[0]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                gen_text = gen_text.split("Paraphrased text :")[1]
            else:
                if lang == 'en':
                    prompt = f"{prompt_en} {ex[:1600]}\n[Paraphrase]:"
                else:
                    prompt = f"{prompt_sv} {ex[:1600]}\n[Omskrivning]:"
                encoded_sent = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048-300)["input_ids"]
                gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=300, temperature=0.7, do_sample=True)[0]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).replace(prompt, "")
                gen_text = gen_text.split('###')[0].strip().replace('\n', ' ')
                gen_text = gen_text.split('[Paraphrase')[0]
                gen_text = gen_text.split('[Original')[0]
                gen_text = gen_text.split('Input:')[0]
                gen_text = gen_text.split('Output:')[0]
                gen_text = gen_text.split('     ')[0]
                gen_text = gen_text.split('\n')[0]
            if gen_text.strip() != '' and gen_text.strip() != ex and gen_text not in ex and len(gen_text) > 150:
                generated_texts.append(gen_text.strip())
        
        if generated_texts:
            query_embedding = model_emb.encode(ex, convert_to_tensor=True)
            corpus_embeddings = model_emb.encode(generated_texts, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)
            print(cosine_scores)
            generated_texts = [t for i, t in enumerate(generated_texts) if 0.65 < cosine_scores[0][i] < 0.99]
        else:
            print(ex)
            print('There was no example similar enough!')
        if generated_texts:
            chosen_texts, chosen_scores = sort_augmented(ex,lab,generated_texts,classifier, num_aug=examples_to_generate)
            augmented_texts += chosen_texts
            augmented_labels += len(chosen_texts) * [lab]
            print(ex)
            print(chosen_texts)
            print(chosen_scores)
        else:
            print(ex)
            print('There was no example similar enough!')
    del model, tokenizer
    if adapter:
        del model_orig, config
        gc.collect()
        torch.cuda.empty_cache()
    return augmented_texts, augmented_labels



###################################################################
# This is where the code for training the classifier starts
###################################################################
def get_train_test_dataset(df):
    df_labeled = df.dropna(subset=['label'])
    df_labeled.reset_index(inplace=True)
    
    if 'split' in df_labeled.columns:
        split_label_list = df_labeled.groupby(['label', 'split'], as_index=False).count()['label'].tolist()
        error_categories = [item for item in split_label_list if split_label_list.count(item) < 2]
        if error_categories:
            raise ValueError('The following categories are not present in both test and train sets:', error_categories)
        else:
            df_test = df_labeled[df_labeled['split'].isin(['Test', 'test', 'Testing', 'testing'])]
    else:
        # get test sample
        label_counts = df_labeled.groupby('label').count()['text'].tolist()
        if 1 in label_counts:
            raise ValueError('You need to have at least 2 examples per category to be able to run the code.')
        else:
            if all(num >= 10 for num in label_counts):
                smallest_num = int(0.2 * min(df_labeled.groupby('label').count()['text'].tolist()))
                df_test = df_labeled.groupby('label', group_keys=False).apply(lambda x: x.sample(smallest_num))
            else:
                df_test = df_labeled.groupby('label', group_keys=False).apply(lambda x: x.sample(1))
                test_temp = df_labeled.drop(index=df_test.index)
                df_test = pd.concat([df_test, test_temp.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.2))])

    # get everything but the test sample
    df_train = df_labeled.drop(index=df_test.index)
   
    return df_train, df_test


def train_final_model(state):
    model_name = MODEL_NAME_ENG
    num_classes = state[0]['num_classes']
    label_to_int_dict = state[0]['label_to_int']
    int_to_label_dict = state[0]['int_to_label']
    df_train = state[0]['train_dataset']
    df_test = state[0]['test_dataset']
    df_train['label'] = df_train.label.apply(lambda x: label_to_int_dict[x])
    df_test['label'] = df_test.label.apply(lambda x: label_to_int_dict[x])
    dataset_train = Dataset.from_pandas(df_train)
    dataset_train = dataset_train.shuffle()
    dataset_test = Dataset.from_pandas(df_test)
    
    def model_init(params):
        params = params or {}
        max_iter = params.get("max_iter", 100)
        solver = params.get("solver", "liblinear")
        params = {
            "head_params": {
            "max_iter": max_iter,
            "solver": solver,
            }
        }
        if num_classes == 2:
            return SetFitModel.from_pretrained(model_name, **params)
        return SetFitModel.from_pretrained(model_name, multi_target_strategy="one-vs-rest", **params)

    def hp_space(trial):  # Training parameters
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_iterations": trial.suggest_categorical("num_iterations", [10, 20, 30, 50]),
            "max_iter": trial.suggest_int("max_iter", 50, 300),
        }


    trainer = SetFitTrainer(
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        model_init=model_init,
        column_mapping={"text": "text", "label": "label"},
    ) 

    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=10)
    print(best_run)
    print(state[0]["results"][-1])
    #Create df
    df_pred_final = pd.DataFrame()
    df_train = pd.concat([df_train, state[0]['df_augmented']]).drop_duplicates(subset = 'text', keep=False)
    df_pred_final['text'] = df_train['text'].tolist() + df_test['text'].tolist() + state[0]['unlabelled_data']['text'].tolist()
    final_result = None
    if len(list(state[0]['unlabelled_data']['text'])) > 0:
        if best_run.objective >= state[0]["results"][-1]:
            trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
            trainer.train()
            metrics = trainer.evaluate()
            final_result = round(100 * metrics['accuracy'])
            pred = trainer.model(list(state[0]['unlabelled_data']['text']))
            pred = [t.item() for t in pred]
            del trainer
        else:
            pred = state[0]["initial_preds"]
    else:
        pred = []
    df_pred_final['label'] = [int_to_label_dict[item] for item in df_train['label'].tolist()] + [int_to_label_dict[item] for item in df_test['label'].tolist()] + [int_to_label_dict[item] for item in pred]
    df_pred_final = pd.merge(df_pred_final, state[0]['original_df'][['project_id', 'text']], on='text', how='left')
    df_pred_final = df_pred_final[['project_id', 'text', 'label']]
    df_pred_final.to_excel("result_sheets/classifier_results.xlsx", index = False)
    if final_result is None:
        final_result = 100 * state[0]["results"][-1]
    return final_result, df_pred_final[:100], "result_sheets/classifier_results.xlsx", state

    
def evaluate(active_learner, test):
    y_pred_test = active_learner.classifier.predict(test)
    test_acc = accuracy_score(y_pred_test, test.y)
    return test_acc


def train_initial_model(file_path, lang, synthetic, state):
    # simulate a warm start
    df = pd.read_excel(file_path.name, engine='openpyxl')
    try:
        df_labelled = df[['project_id', 'text', 'label', 'split']]
    except KeyError:
        df_labelled = df[['project_id', 'text', 'label']]
    df_labelled = df_labelled.dropna(subset=['project_id', 'text', 'label'])
    df_unlabelled = df[df['label'].isna()]
    df_unlabelled = df_unlabelled.dropna(subset=['text'])
    df_unlabelled = df_unlabelled[['project_id', 'text', 'label']]
    df_augmented = pd.DataFrame()
    num_classes = len(df_labelled['label'].unique())
    label_to_int_dict = {}
    int_to_label_dict = {}
    for i, lab in enumerate(df_labelled['label'].unique()):
        label_to_int_dict[lab] = i
        int_to_label_dict[i] = lab
    print(label_to_int_dict)
    target_labels = np.arange(num_classes)
    dataset_train, dataset_test = get_train_test_dataset(df_labelled)
    #Add synthetic data
    if synthetic == 'Add synthetic examples':
        df_train_aug = copy.deepcopy(dataset_train)
        df_test_aug = copy.deepcopy(dataset_test)
        df_train_aug['label'] = df_train_aug.label.apply(lambda x: label_to_int_dict[x])
        df_test_aug['label'] = df_test_aug.label.apply(lambda x: label_to_int_dict[x])
        dataset_train_aug = Dataset.from_pandas(df_train_aug)
        dataset_train_aug = dataset_train_aug.shuffle()
        dataset_test_aug = Dataset.from_pandas(df_test_aug)
        
        # Load a SetFit model from Hub
        if lang == 'EN':
            model_aug = SetFitModel.from_pretrained(MODEL_NAME_ENG)
        else:
            model_aug = SetFitModel.from_pretrained(MODEL_NAME_SV)

        # Create trainer
        trainer_aug = SetFitTrainer(
                   model=model_aug,
                   train_dataset=dataset_train_aug,
                   eval_dataset=dataset_test_aug,
                   loss_class=CosineSimilarityLoss,
                   metric="accuracy",
                   batch_size=16,
                   num_iterations=30, # The number of text pairs to generate for contrastive learning
                   num_epochs=1, # The number of epochs to use for contrastive learning
                   column_mapping={"text": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
                   )

        # Train and evaluate
        trainer_aug.train()
        
        augmented_texts, augmented_labels = get_augmented_data(df_train_aug['text'].tolist(), df_train_aug['label'].tolist(), trainer_aug.model, lang=lang.lower(), examples_to_generate=1)
        del dataset_train_aug, dataset_test_aug, model_aug, trainer_aug
        df_augmented['text'] = augmented_texts
        df_augmented['label'] = [int_to_label_dict[item] for item in augmented_labels]
    
    
    dataset_train = dataset_train.append(df_augmented)
    train_texts = list(dataset_train['text']) + df_unlabelled['text'].tolist()
    train_labels = [label_to_int_dict[item] for item in dataset_train['label']] + df_unlabelled.shape[0] * [LABEL_UNLABELED]
    

    train = TextDataset.from_arrays(train_texts, np.array(train_labels), target_labels = target_labels)
    test = TextDataset.from_arrays(list(dataset_test['text']), np.array([label_to_int_dict[item] for item in dataset_test['label']]),
                                  target_labels = target_labels)
    unl = TextDataset.from_arrays(df_unlabelled['text'].tolist(), np.array(df_unlabelled.shape[0] * [LABEL_UNLABELED]), target_labels = target_labels)
    
    if lang == 'EN':
        sentence_transformer_model_name = MODEL_NAME_ENG
    else:
        sentence_transformer_model_name = MODEL_NAME_SV
    setfit_model_args = SetFitModelArguments(sentence_transformer_model_name)
    clf_factory = SetFitClassificationFactory(setfit_model_args, num_classes)

    # define a query strategy and initialize a pool-based active learner
    query_strategy = SubsamplingQueryStrategy(BreakingTies())
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    ind = np.array(list(range(len(list(dataset_train['text'])))))
    active_learner.initialize_data(ind, np.array([label_to_int_dict[item] for item in dataset_train['label'].tolist()]))
    labeled_indices = ind

    results_setfit = []
    result = evaluate(active_learner, test)
    results_setfit.append(result)
    
    
    y_pred_train = active_learner.classifier.predict(train)
    #y_pred_unl = active_learner.classifier.predict(unl)
    #print(y_pred_unl)
    df_predict = pd.DataFrame(columns=['text', 'label'])
    state = [{'test_dataset' : dataset_test, 'train_dataset' : dataset_train, 'unlabelled_data' : df_unlabelled, 'active_learner' : active_learner, 'results' : results_setfit, 'num_classes' : num_classes, 'label_to_int' : label_to_int_dict, 'int_to_label' : int_to_label_dict, 'last_queried_indices' : [], 'train' : train, 'test' : test, 'original_df' : df, 'labeled_indices' : labeled_indices, 'train_texts' : train_texts, 'df_augmented' : df_augmented, 'lang' : lang, 'initial_preds' : y_pred_train[len(list(dataset_train['text'])):], 'target_labels' : target_labels}]
    if df_unlabelled.shape[0] > 0:
        num_samples = min(10*num_classes, int(len(df_unlabelled['text'].tolist())/2))
        q_indices = active_learner.query(num_samples=num_samples)
        predicted_texts_labels = [(item[0], item[1]) for i, item in enumerate(zip(train_texts, y_pred_train)) if i in q_indices]
        state[0]['last_queried_indices'] = q_indices
    
        #Create df
        df_predict['text'] = [item[0] for item in predicted_texts_labels]
        df_predict['label'] = [int_to_label_dict[item[1]] for item in predicted_texts_labels]
        df_predict = pd.merge(df_predict, df[['project_id', 'text']], on='text', how='left')
        df_predict = df_predict[['project_id', 'text', 'label']]
    return round(100 * result), df_predict, state


def continue_training_model(df_predict, state):
        label_to_int_dict = state[0]['label_to_int']
        int_to_label_dict = state[0]['int_to_label']
        active_learner = state[0]['active_learner']
        q_indices = state[0]['last_queried_indices']
        labeled_indices = state[0]['labeled_indices']       

        if set(list(df_predict['text'])).issubset(state[0]['train_dataset']['text']):
            return 100 * state[0]['results'][-1], df_predict, state

        df_unlabelled_new = pd.concat([state[0]['unlabelled_data'], df_predict]).drop_duplicates(subset = 'text', keep=False)
        dataset_train = state[0]['train_dataset'].append(df_predict)
        state[0]['unlabelled_data'] = df_unlabelled_new
        state[0]['train_dataset'] = dataset_train  

        unl = TextDataset.from_arrays(list(df_unlabelled_new['text']), np.array(df_unlabelled_new.shape[0] * [LABEL_UNLABELED]), target_labels = state[0]['target_labels'])
        updated_initial_preds = active_learner.classifier.predict(unl)
        print(updated_initial_preds)
        state[0]['initial_preds'] = updated_initial_preds
        
        if df_unlabelled_new.shape[0] < 10:
            return round(100 * state[0]['results'][-1]), df_predict, state
        
        #Get new indices
        y = np.array([label_to_int_dict[item] for item in list(df_predict['label'])])

        # Return the labels for the current query to the active learner.
        active_learner.update(y)
    
        # memory fix: https://github.com/UKPLab/sentence-transformers/issues/1793
        gc.collect()
        torch.cuda.empty_cache()
        
        #Predict train
        y_pred_train = active_learner.classifier.predict(state[0]['train'])
        result = evaluate(active_learner, state[0]['test'])
        state[0]['results'].append(result)
        num_samples = min(10*state[0]['num_classes'], int(len(state[0]['unlabelled_data']['text'].tolist())/2))
        q_indices = active_learner.query(num_samples=num_samples)
        predicted_texts_labels = [(item[0], item[1]) for i, item in enumerate(zip(state[0]['train_texts'], y_pred_train)) if i in q_indices]
        state[0]['last_queried_indices'] = q_indices
        labeled_indices = np.concatenate([q_indices, labeled_indices])
        state[0]['labeled_indices'] = labeled_indices
        
        #Create df_predict
        df_predict = pd.DataFrame()
        df_predict['text'] = [item[0] for item in predicted_texts_labels]
        df_predict['label'] = [int_to_label_dict[item[1]] for item in predicted_texts_labels]
        df_predict = pd.merge(df_predict, state[0]['original_df'][['project_id', 'text']], on='text', how='left')
        df_predict = df_predict[['project_id', 'text', 'label']]
        
        return round(100 * result), df_predict, state

    