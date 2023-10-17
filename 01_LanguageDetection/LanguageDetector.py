from googletrans import Translator
import pandas as pd
import spacy

translator = Translator()
nlp = spacy.load("en_core_web_trf")
df = pd.read_excel("BBisenglish15000.xlsx")
print('import done')


x = 15000
for i, ticket in enumerate(df.loc[x:, 'cleaned_string']):
    mix_dic = {}
    is_english = True
    doc = nlp(ticket)
    # remove the names that effect the detection
    names = [i for i in doc.ents if i.label_.lower() in ["person"]]
    for name in names:
        ticket = ticket.replace(str(name), '')
    doc = nlp(ticket)
    sentences = [sent.text.strip() for sent in doc.sents]
    sentences = list(filter(None, sentences))
    if len(sentences) >= 1:
        for sent in sentences:
            detection = translator.detect(sent)
            if isinstance(detection.lang, str):
                detected_lan = detection.lang
                detected_con = detection.confidence
                try:
                    mix_dic[detected_lan] += detected_con
                except:
                    mix_dic[detected_lan] = detected_con
            elif isinstance(detection.lang, list):
                for lang_i in range(len(detection.lang)):
                    detected_lan = detection.lang[lang_i]
                    detected_con = detection.confidence[lang_i]
                    try:
                        mix_dic[detected_lan] += detected_con
                    except:
                        mix_dic[detected_lan] = detected_con

        eng_lan = 0
        other_lan = 0
        for k, v in mix_dic.items():
            if v <= 0.7:
                mix_dic[k] = 0
            else:
                if k == 'en':
                    eng_lan = v
                else:
                    other_lan += v

        if eng_lan > 0:
            if eng_lan / (eng_lan + other_lan) >= 0.75:
                is_english = True
            else:
                is_english = False
        else:
            is_english = False

    if is_english:
        df.loc[i+x, 'isEnglish'] = 1
    else:
        df.loc[i+x, 'isEnglish'] = 0

    if (i+x) > x:
        if (i+x) % 1000 == 0:
            print((i+x))
        if (i+x) % 10000 == 0:
            df.to_excel(f'BBisenglish{i+x}.xlsx')

df.to_excel('cleanBBR2021.xlsx')
