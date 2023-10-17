import spacy

nlp = spacy.load('en_core_web_lg')


def document_to_LSTM_example(doc):
    '''
    This takes a document Object and creates an example for the training data set
    :param doc: object from mongodb
    :return: (splitted sentence as array, tags for the splitted sentence as array)
    '''
    sentence = doc['question']
    nFormat = doc['markupNFormat']

    token_array = []
    tag_array = []
    finalResult = []

    doc = nlp(sentence)

    for token in doc:
        token_array.append(token.lemma_.lower())
        found = False
        for field in nFormat:
            if (field['start'] <= token.idx and field['end'] > token.idx and token.text != '\n'):
                found = True
                tag_array.append(field['key'])
        if found == False:
            tag_array.append('irrelevant')
    #for i in range(len(token_array)):
    #    field= (token_array[i], tag_array[i])
    #    finalResult.append(field)

    #return finalResult
    return token_array, tag_array
