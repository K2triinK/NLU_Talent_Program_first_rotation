# NLU_Talent_Program_first_rotation

Part of the work done during the first rotation of NLU Talent Program between October 2022 and June 2023. Code that did not include any confidential information.

PEFT adapters are missing, so trying to use them will not work. Also, GPT-SW3 models can only be used if you have gained access to them.

## Classify documents
The 'classify documents' part of the app is linked to the ```classifier_setfit.py``` script. The main idea is that the user can insert an Excel-file with at least 10 labelled examples per category and leave the rest of the label column blank. Now, an initial model will be trained and the user will be shown example predictions (10 per label). Those are selected using small_text, an active learning library, based on uncertainty. If the user is happy with the predictions, they can choose to train a final model (hyperparameter search is carried out here) and get labels for all unlabelled examples. Otherwise, the user can choose to correct the mistakes in the example predictions and add these examples to the training data and continue with that until they are happy with the example predictions and ready to proceed to training the final model. There is also an option to add augmented training examples in which case AI Sweden's GPT-SW3 is used to paraphrase the existing examples to increase the amount of training data.

## Discover topics
The 'Discover topics' part of this app is linked to the ```discover_topics.py``` script. This functionality allows the user to automatically find out what topics are present in a specific population of documents and obtain a visualization of the topics with their headings or titles. To perform this task, the user needs to provide the app with an excel file in the same format as the output from 'Find documents by keywords', 'Find relevant documents' and 'Find relevant documents', and is asked to specify the language of the source texts (either English, 'EN', or Swedish, 'SV'), how many different group topics they want to see (the maximum is 20 and, although the app allows lower, we recommend to ask for at least 2 groups), and whether they want to allow or disallow outliers. If 'Allow outliers' is picked, the model can to leave out any project that does not fit any group, otherwise, all projects will be assigned a topic.

With the provided excel file, POS tags and lemmas are extracted from the set of documents and keeps those that are nouns, verbs, proper nouns and adjectives. It then determines the number of neighbors and components and creates a umap model (a non-linear dimension reduction algorithm) measured by cosine similarity. After that, the cluster model is made, with KMeans if outliers are disallowed, and with HBDSCAN if outliers are allowed. Finally, topic modeling is performed with BERTopic and topic headings are generated for each of the topics by prompting AI Sweden's GPT-SW3 6.7B v2 model.

## Map to topics
The 'Map to topics' part of the app is linked to the ```map_to_topics.py``` script. This feature allows the user to map or classify a population of documents into their desired topics or classes, and obtain a visualization of the topic distribution. To do so, the user is initially asked to provide at least two topic titles and a summary of the topic or an example text, as well as an excel file of the projects that they want to map in the same format as the output from 'Find documents by keywords', 'Find relevant documents' and 'Find relevant documents', and they will have to specify the input language.

With the provided excel file and topic titles and descriptions, the topics and the projects are embedded and cosine similarities are computed, and a UMAP is built to then obtain the 2D representations of the embeddings on the x and y axis. This information is then combined and a visualization with the projects distributed in the axis and grouped according to the provided topics is displayed.


## Extract keywords
The 'Extract keywords' part of the app is linked to the ```get_keywords.py``` script. When provided with a project, this feature will extract ten keywords that define the text. This is done by prompting a Parameter-Efficient Fine-Tuning (PEFT) tuned version of the GPT-SW3 6.7B by AI Sweden. The keywords are then parsed from the model output and checked for duplicates before being shown to the user.


## The following apps are experimental and therefore unstable, you may or may not get a good result ##

## Ask PDFs
The 'Ask PDFs' part of this app is linked to the ```ask_docs.py``` script. In this functionality, the user can directly ask questions related to PDF files placed in the "pdf_documents" folder. First, semantic search is used to extract the 5 most relevant documents/paragraphs to the question. Then, GPT-SW3 is used for generating the answer. There are two prompting 'phases' where the first is fewshot prompting following a 'background-question-answer' format, and the second, called 'refining', starts with a prompt that says "Change the original answer if the background text contains new information relevant to the question", and then follows a 'original question-current answer-background-refined answer' format.

## Zero-shot classification
The 'Zero-shot classification' part of the app is linked to the ```zeroshot_classifier.py``` script. This feature allows the end user to classify one project text into their manually specified categories. This is done with zero-shot prompting (so, giving the model no previous examples of the task) with a Parameter-Efficient Fine-Tuning (PEFT) tuned version of the GPT-SW3 6.7B by AI Sweden. The output is the first category that the model generates.
