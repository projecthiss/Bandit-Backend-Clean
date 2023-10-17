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
                tag_array.append(field['key'].upper())
        if found == False:
            tag_array.append('Unrelevant'.upper())
    #for i in range(len(token_array)):
    #    field= (token_array[i], tag_array[i])
    #    finalResult.append(field)

    #return finalResult
    return token_array, tag_array

#a,b=document_to_LSTM_example({"_id":{"$oid":"629344326d4d3b4b08bf934d"},"includedForSolutions":False,"hasBeenReviewed":True,"recommendedTickets":[],"deleted":False,"markupNFormat":[{"_id":{"$oid":"6293575e0b4d29002c933b37"},"start":152,"end":187,"color":"#82b1ff","key":"fehlerbeschreibung"},{"_id":{"$oid":"6293575e0b4d29002c933b38"},"start":108,"end":121,"color":"#82b1ff","key":"fehlerbeschreibung"},{"_id":{"$oid":"6293575e0b4d29002c933b39"},"start":86,"end":91,"color":"#82b1ff","key":"fehlerbeschreibung"},{"_id":{"$oid":"6293575e0b4d29002c933b3a"},"start":62,"end":73,"color":"#ffd180","key":"system"},{"_id":{"$oid":"6293575e0b4d29002c933b3b"},"start":45,"end":56,"color":"#82b1ff","key":"fehlerbeschreibung"}],"internID":82552,"uhd_NR":72296,"date":{"$date":"2019-10-21T22:00:00.000Z"},"report":"Problem beim Fileexport bei den Firmen Rotus und HMS","category_1":"SW DMS Starke","category_2":"Fileexport","question":"Guten Morgen Herr Dilling,\nes scheint irgend ein Problem beim Fileexport zu geben.\nEs fand gestern um 13 Uhr keiner statt und bei dem um 23 Uhr wurden scheinbar nicht alle Daten übergeben.\nBin im Moment etwas ratlos.\nVielleicht könnten Sie sich bei mir melden.\nMit freundlichen Grüßen\ni.A. Andre Bierwirth\nBuchhaltung\nHERMANNS HTI-Bau GmbH u. Co. KG\nWilhelm-Speck-Str. 17\n34125 Kassel\nDeutschland\nTelefon +49 (561) 8792-446\nFax +49 (561) 8792-497\nE-Mail bierwirth@hermanns.de <mailto:bierwirth@hermanns.de>\nInternet www.hermanns.de\nKommanditgesellschaft HRA 7346 Kassel\npers. haft. Gesellschafterin:\nHERMANNS HTI-Bau und Geschäftsführungs GmbH\nAmtsgericht Kassel HRB 3468, USt.-Id.-Nr.: DE 113024495\nGeschäftsführer: Hans-Ulrich Hujer, Swen Haar\nEin Unternehmen der HERMANNS-Gruppe.\nSofern nicht anders angegeben, ist diese E-Mail und alle Anlagen vertraulich und nur für den Empfänger bestimmt.\nSollten Sie diese E-Mail irrtümlich erhalten haben, ist Ihnen eine Kenntnisnahme des Inhaltes, eine Vervielfältigung\noder Weitergabe der E-Mail ausdrücklich untersagt. Bitte benachrichtigen Sie uns und vernichten Sie die empfangene E-Mail.\n","answerhistory":[{"_id":{"$oid":"629344336d4d3b4b08bf934e"},"date":{"$date":"2019-11-23T09:07:01.000Z"},"content":"Dilling, Jürgen: Verarbeitungsstatus \"Meldung erledigt\" gesetzt.\r\r\n\r\r\n"},{"_id":{"$oid":"629344336d4d3b4b08bf934f"},"date":{"$date":"2019-11-23T09:06:53.000Z"},"content":"Dilling, Jürgen: Telefonisch erledigt.\r\r\nLag daran, dass Herr Bierwirth nicht alle Belege auf das gestern neu eingeführte Kennzeichen: Einzel/Aufteilungsbuchung umgestellt hatte.\r\r\n\r\r\n"},{"_id":{"$oid":"629344336d4d3b4b08bf9350"},"date":{"$date":"2019-11-23T09:06:52.000Z"},"content":"Dilling, Jürgen: Verarbeitungsstatus \"Meldung erfasst\" gesetzt.\r\r\n"}],"evaluation":False,"clusterCat":"NULL","preSolutionStandard":False,"userShownID":13615,"__v":0,"markupTFormat":{"fehlerbeschreibung":["ein Problem","fand "," keiner statt","cheinbar nicht alle Daten übergeben"],"system":["Fileexport "]},"reviewedDate":{"$date":"2022-05-29T11:22:06.117Z"}})

#for i in range(len(a)):
#    print(b[i]+" : "+a[i])