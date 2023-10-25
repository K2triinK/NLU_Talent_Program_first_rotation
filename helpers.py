from easynmt import EasyNMT
import langid

langid.set_languages(['sv', 'en'])


def trans(txt, tgt_lang='en'):
    # Translates input text into English, without requiring the user to specify the source language through neural machine translation
    model = EasyNMT('opus-mt', max_loaded_models=5)
    src_lang, _ = langid.classify(txt)
    translated = model.translate(txt, source_lang=src_lang, target_lang=tgt_lang)
    return translated