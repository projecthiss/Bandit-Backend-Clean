import spacy

nlp = spacy.load('en_core_web_lg')


def document_to_BERT_example(doc):
    '''
    This takes a document Object and creates an example for the training data set
    :param doc: input ticket
    :return: (splitted sentence as array, tags for the splitted sentence as array)
    '''

    token_array = []

    doc = nlp(doc)

    for token in doc:
        token_array.append(token.lemma_.lower())

    return token_array

