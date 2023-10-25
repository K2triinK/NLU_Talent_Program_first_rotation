from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextGenerationPipeline
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import yaml
#from unstructured.documents.elements import NarrativeText
#from unstructured.partition.text_type import sentence_count
from langchain import PromptTemplate, FewShotPromptTemplate
from peft import PeftModel, PeftConfig
import torch
os.environ["UNSTRUCTURED_LANGUAGE_CHECKS"] = "true"

credentials = yaml.load(open('./credentials.yml'),  Loader=yaml.FullLoader)
AUTH_TOKEN = credentials['credentials']['auth_token']


class MyPipeline(TextGenerationPipeline):
    def preprocess(self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs):
        inputs = self.tokenizer(
            prefix + prompt_text, padding=False, add_special_tokens=False, return_tensors=self.framework
        )
        inputs["prompt_text"] = prompt_text
        input_ids = [3, *self.tokenizer(f'User: {prompt_text}')['input_ids'], 2, *self.tokenizer('Bot:')['input_ids']]
        input_ids = torch.tensor(input_ids).to("cuda:0").unsqueeze(0)
        inputs["input_ids"] = input_ids
        del inputs['token_type_ids']
        del inputs['attention_mask']
        return inputs

def ask_docs(q, adapter=True):
    examples_question = [
    {"bakgrund": "Senast år 2045 ska Sverige inte ha några utsläpp som skadar klimatet. Det handlar om utsläpp från till exempel bilar och fabriker. Men för första gången på 20 år ökar Sveriges utsläpp. Och regeringens politik kommer göra att utsläppen fortsätter att öka. Det säger experterna i Klimatpolitiska rådet. Rådet jobbar med att undersöka regeringens politik för klimatet. Om regeringen inte ändrar sin politik går det åt helt fel håll för Sverige. Det säger Cecilia Hermansson som leder rådet.", "fråga": "Varför får den svenska regeringen kritik?", "svar" : "Eftersom regeringens politik gör att Sveriges utsläpp kommer att öka istället för att minska."},
    {"bakgrund": "Bilar som använder bensin och diesel släpper ut gaser som skadar klimatet. Nu ska EU stoppa sådana bilar. De får inte säljas efter år 2035. Ministrar för miljön i alla länder i EU bestämde det på tisdagen. 23 länder i EU röstade ja. Tre länder ville inte rösta. Bara ett land röstade nej. Det var Polen. Sverige röstade ja.", "fråga": "Vilka länder ville inte rösta?", "svar" : "Jag vet inte för det nämns inte i bakgrunden."},
    {"bakgrund": "Fransmännen är arga på sin regering. Politikerna ska höja lägsta åldern för när folk får gå i pension från 62 år till 64 år. Folk har protesterat flera gånger mot det. I tisdags gick människor ut och protesterade igen. De flesta som protesterar har inte varit våldsamma. Men några har varit våldsamma mot poliser. Flera hundra poliser har blivit skadade under protesterna.", "fråga": "Vad handlar protesterna i Frankrike om?", "svar" : "Lägsta pensionsåldern som kommer ökas från 62 till 64 år."}
    ]

    # Next, we specify the template to format the examples we have provided.
    # We use the `PromptTemplate` class for this.
    example_formatter_template_question = """
    Bakgrund: {bakgrund}
    Fråga: {fråga}
    Svar: {svar}"""

    example_prompt_question = PromptTemplate(
        input_variables=["bakgrund", "fråga", "svar"],
        template=example_formatter_template_question,
    )

    # Finally, we create the `FewShotPromptTemplate` object.
    few_shot_prompt_question = FewShotPromptTemplate(
        # These are the examples we want to insert into the prompt.
        examples=examples_question,
        # This is how we want to format the examples when we insert them into the prompt.
        example_prompt=example_prompt_question,
        # The prefix is some text that goes before the examples in the prompt.
        # Usually, this consists of intructions.
        prefix="Svara på frågan med hjälp av informationen i bakgrunden och inte baserat på tidigare kunskap.",
        # The suffix is some text that goes after the examples in the prompt.
        # Usually, this is where the user input will go
        suffix="""
        Bakgrund: {context_str}
        Fråga: {question}
        Svar:""",
        # The input variables are the variables that the overall prompt expects.
        input_variables=["context_str", "question"],
        # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
        example_separator="\n###\n",
    )
    
    
    examples_refine = [{'originalfråga' : 'Varför får den svenska regeringen kritik?', 'nuvarande' : 'Eftersom regeringens politik gör att Sveriges utsläpp kommer att öka istället för att minska.', 'bakgrund' : 'Bilar som använder bensin och diesel släpper ut gaser som skadar klimatet. Nu ska EU stoppa sådana bilar. De får inte säljas efter år 2035. Ministrar för miljön i alla länder i EU bestämde det på tisdagen. 23 länder i EU röstade ja. Tre länder ville inte rösta. Bara ett land röstade nej. Det var Polen. Sverige röstade ja.', 'förfinat' : 'Eftersom regeringens politik gör att Sveriges utsläpp kommer att öka istället för att minska.'},
    {'originalfråga' : 'Vad handlar protesterna i Frankrike om?', 'nuvarande' : 'Pensionen.', 'bakgrund' : 'Fransmännen är arga på sin regering. Politikerna ska höja lägsta åldern för när folk får gå i pension från 62 år till 64 år. Folk har protesterat flera gånger mot det. I tisdags gick människor ut och protesterade igen. De flesta som protesterar har inte varit våldsamma. Men några har varit våldsamma mot poliser. Flera hundra poliser har blivit skadade under protesterna.', 'förfinat' : 'Lägsta pensionsåldern som kommer ökas från 62 till 64 år.'}
]

    # Next, we specify the template to format the examples we have provided.
    # We use the `PromptTemplate` class for this.
    example_formatter_template_refine = """
    Originalfråga: {originalfråga}
    Nuvarande svar: {nuvarande}
    Bakgrund: {bakgrund}
    Förfinat svar: {förfinat}"""

    example_prompt_refine = PromptTemplate(
        input_variables=["originalfråga", "nuvarande", "bakgrund", "förfinat"],
        template=example_formatter_template_refine,
    )

    # Finally, we create the `FewShotPromptTemplate` object.
    few_shot_prompt_refine = FewShotPromptTemplate(
        # These are the examples we want to insert into the prompt.
        examples=examples_refine,
        # This is how we want to format the examples when we insert them into the prompt.
        example_prompt=example_prompt_refine,
        # The prefix is some text that goes before the examples in the prompt.
        # Usually, this consists of intructions.
        prefix="Förfina nuvarande svaret ifall bakgrunden innehåller ny och relevant information som svarar på frågan.",
        # The suffix is some text that goes after the examples in the prompt.
        # Usually, this is where the user input will go
        suffix="""
        Originalfråga: {question}
        Nuvarande svar: {existing_answer}
        Bakgrund: {context_str}
        Förfinat svar:""",
        # The input variables are the variables that the overall prompt expects.
        input_variables=["question", "existing_answer", "context_str"],
        # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
        example_separator="\n###\n",
    )
    
    refine_template = (
    "Change the original answer if the background text contains new information relevant to the question.\nQuestion: {question}\nCurrent answer: {existing_answer}\nBackground: {context_str}\nEnhanced answer:"
    )

    refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_template,
    )

    question_template = (
    "Answer the question based on the context.\nContext: {context_str}\nQuestion: {question}\nAnswer:"
    )

    question_prompt = PromptTemplate(
        input_variables=["context_str", "question"], template=question_template
    )
    
    loader = DirectoryLoader("./pdf_documents/") 
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='KBLab/sentence-bert-swedish-cased')
    
    db = Chroma.from_documents(docs, embeddings)
    query = q
    docs = db.similarity_search(query, k=5)
    print(docs)
    
    if adapter:
        model_name = "AI-Sweden/gpt-sw3-6.7b-private"
        peft_model_id = "./adapters/qa_multitask_v1"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto', use_auth_token=AUTH_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=AUTH_TOKEN)
        model = PeftModel.from_pretrained(model, peft_model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400, temperature=0)
        hf = HuggingFacePipeline(pipeline=pipe)
        chain = load_qa_with_sources_chain(hf, chain_type="refine", return_intermediate_steps=True, question_prompt=question_prompt, refine_prompt=refine_prompt)
    else:
        model_name = "AI-Sweden/gpt-sw3-6.7b-v2-instruct-private" #"circulus/alpaca-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=AUTH_TOKEN, truncation=True, max_length=1648)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400, temperature=0, eos_token_id=[2,3], pipeline_class=MyPipeline)
        hf = HuggingFacePipeline(pipeline=pipe)
        chain = load_qa_with_sources_chain(hf, chain_type="refine", return_intermediate_steps=True, question_prompt=few_shot_prompt_question, refine_prompt=few_shot_prompt_refine)   

    output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)['output_text']
    
    return output