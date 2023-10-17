from transformers import TFAutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline



tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer/", repo_type="models")
model = TFAutoModelForTokenClassification.from_pretrained("trained_bert", local_files_only=True)

token_classifier = pipeline(
    "token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="first"
)

def get_bert_highlighting(sentence):
    try:
        pred = token_classifier(sentence)


        t_Format = {"system":[], "faultdescription":[], "source": [], "servicerequest": [], "other": [] }
        n_Format = []
        i=0
        for element in pred:

            tag = element["entity_group"]
            if tag != "irrelevant":
               t_Format[tag.lower()].append(element["word"])
               if tag == "system":
                   #print("System reached")
                   n_Format.append({"key": "system", "start": element["start"],
                                    "end":  element["end"]})
               elif tag == "faultdescription":
                   n_Format.append({ "key": "faultdescription", "start":  element["start"],
                                    "end": element["end"]})
               elif tag == "source":
                   n_Format.append({ "key": "source", "start":  element["start"],
                                    "end": element["end"]})
               elif tag == "servicerequest":
                   n_Format.append({ "key": "servicerequest", "start": element["start"],
                                    "end": element["end"]})
               elif tag == "other":
                   n_Format.append({ "key": "other", "start": element["start"],
                                    "end": element["end"]})

            i+=1
        return n_Format, t_Format
    except:
        print("error")
        t_Format = {"system": [], "faultdescription": [], "source": [], "servicerequest": [], "other": []}
        n_Format = []
        return n_Format, t_Format

n, t = get_bert_highlighting("My password needs to be reset.")
print(t)