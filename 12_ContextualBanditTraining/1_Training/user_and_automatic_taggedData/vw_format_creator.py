import re
import spacy

# PARAMS
NUMBER_OF_SLOTS = 1


def set_slot_number(i):
    global NUMBER_OF_SLOTS
    NUMBER_OF_SLOTS = i


import re
import spacy
import json

nlp = spacy.load("de_core_news_lg")


def createActionsList(all_actions, included_items=None):
    action_list = ""
    i = 0
    if included_items is None:
        included_items = len(all_actions)
    while i < included_items:
        local_action = "ccb action" + createContext(all_actions[i])
        action_list = action_list + local_action
        i += 1
    return action_list


def createContext(item):
    context = ""
    t = json.dumps(item)

    for c in json.loads(t):
        if c != 'context':
            context += " |" + c
            item[c] = set(item[c])
            item[c] = list(item[c])
            i = 0
            while i < len(item[c]):
                item[c][i] = item[c][i].replace("|", "")
                item[c][i] = item[c][i].strip()
                item[c][i] = item[c][i].replace("\n", " ")
                item[c][i] = item[c][i].replace("\r", " ")
                item[c][i] = item[c][i].replace("  ", " ")
                item[c][i] = item[c][i].replace("  ", " ")
                item[c][i] = item[c][i].replace(":", "")
                item[c][i] = re.sub("\s\s+", " ", item[c][i])
                i = i + 1
            for content in item[c]:
                for tok in nlp(content):
                    context += " " + tok.lemma_.lower()
    return context + "\n"


def getSlots(all_actions_id, all_actions, taken_actions=None, included_items=None):
    counter = 0
    slots = ""
    if included_items is None:
        included_items = len(all_actions)
    while counter < 4 and counter < included_items:
        if taken_actions is None:
            slots += "ccb slot |\n"
        else:
            slots += "ccb slot " + createTakenActionString(all_actions_id, taken_actions[counter]) + " |\n"
        counter += 1
    return slots


def createTakenActionString(all_actions_id, action):
    id = all_actions_id.index(action['id'])
    return str(id) + ":" + str(action['cost']) + ":" + str(action['probability'])


def createRanking(all_actions_id, pmf):
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Conditional-Contextual-Bandit#prediction-type
    ranking = []
    slot = 0
    while slot < 4 and slot < len(pmf):
        ranking.append({"id": all_actions_id[pmf[slot][0][0]], "probability": pmf[slot][0][1]})
        slot += 1
    return ranking
