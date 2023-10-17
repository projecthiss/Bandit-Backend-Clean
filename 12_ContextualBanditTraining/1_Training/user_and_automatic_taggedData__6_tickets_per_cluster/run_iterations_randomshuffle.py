import random
import vw_format_creator
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sn
import vowpalwabbit
from highlight_with_bert import get_bert_highlighting

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import pickle

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['figure.dpi'] = 1200

all_data = pd.read_excel("output_clustering_and_quality_scoring.xlsx")
my_data = json.loads(all_data.to_json(orient="records"))
print(my_data[0])
# lade daten, welche gemäß der txt Datei überarbeitet wurden sind
with open("all_highlighted_tickets.json", "r", encoding="utf8") as f:
    pre_my_data = json.load(f)

# Ersetze ClusterCats mit meinen clusterCats
# my_cluster = pd.read_excel("my_clust.xlsx")
# for ticket in my_data:
#    for index, row in my_cluster.iterrows():
#        if ticket["uhd_NR"] == row["uhd_NR"] and ticket["internID"] == row["internID"]:
#            ticket["clusterCat"] = row["MyClust"]
#            break
#
## entferne leere cluster
# my_data = [x for x in my_data if type(x["clusterCat"]) != float]

loadit=0
loadfound=0
for ticket in my_data:
    found= False
    if loadit%500==0:
        print(loadit)
    loadit+=1
    for pre_ticket in pre_my_data:
        if pre_ticket["uhd_NR"] == ticket["number"]:
            ticket["markupTFormat"]=pre_ticket["markupTFormat"]
            ticket["clean_solution"]=pre_ticket["question"]
            loadfound+=1
            print(loadfound)
            found=True
            break;
    if not found:
        n_format, t_Format = get_bert_highlighting(ticket["clean_solution"])
        ticket["markupTFormat"]=t_Format
    ticket["clusterCat"]=ticket["merged_clusters"]
    ticket["contextString"] = vw_format_creator.createContext(ticket["markupTFormat"])
    ticket['positiveRewards'] = 0
del pre_my_data


eval_data = []
eval_size = 0.2
eval_ticket_number = eval_size * len(my_data)
while len(eval_data) < eval_ticket_number:
    found = False
    while not found:
        randomNr = random.randint(0, len(my_data) - 1)
        i = 0
        count = 0
        for t in my_data:
            if i != randomNr and my_data[randomNr]["clusterCat"] == t["clusterCat"]:
                count += 1
            i += 1
        if count > 2:
            eval_data.append(my_data.pop(randomNr))
            found = True

LEARN = True
if LEARN:
    vw = vowpalwabbit.pyvw.vw("--ccb_explore_adf --interactions :: --epsilon 0.25")
else:
    vw = vowpalwabbit.pyvw.vw("-i vw_model_from_json_in_iterations_randomshuffle.vow")
    print("Model Loaded")

# lerne alle "nicht actions"
average_pred_value = [0]
highest_pred_value = [0]
cumulated_average_highest_pred_value = [0]
cumulated_average_pred = [0]
iterations = [0]
pred_value_iteration = [[0]]
iteration = 0
NUMBER_OF_SLOTS = 1
vw_format_creator.set_slot_number(NUMBER_OF_SLOTS)
actions = ""
lowestPositiveReward = 0
rewards_zero = len(my_data)

stop_lowest_positive_Reward = 1

with open("train_data.json", "w", encoding="utf8") as f:
    json.dump(my_data, f)
a = pd.read_json("train_data.json")
a.to_excel("train_data.xlsx")
with open("eval_data.json", "w", encoding="utf8") as f:
    json.dump(eval_data, f)
a = pd.read_json("eval_data.json")
a.to_excel("eval_data.xlsx")

testCounter = 0
ccb_string = ""
if (False):
    while testCounter < len(my_data):
        shared_example = "ccb shared " + my_data[testCounter]["contextString"]
        actions_example = ""
        count = 0
        slot_learn = ""
        while count < len(my_data):
            if count != testCounter:
                actions_example += "ccb action" + my_data[count]["contextString"]
                if my_data[testCounter]["clusterCat"] == my_data[count]["clusterCat"]:
                    if count > testCounter:
                        slot_learn += "ccb slot " + str(count + 1) + ":-1:0.75 |\n"
                    else:
                        slot_learn += "ccb slot " + str(count) + ":-1:0.75 |\n"
                else:
                    if count > testCounter:
                        slot_learn += "ccb slot " + str(count + 1) + ":0:0.75 |\n"
                    else:
                        slot_learn += "ccb slot " + str(count) + ":0:0.75 |\n"

            count += 1
        ccb_string += shared_example + actions_example + slot_learn + "\n"
        l = vw.predict(shared_example + actions_example + slot_learn)
        vw_format = vw.parse(shared_example + actions_example + slot_learn)
        vw.learn(vw_format)
        testCounter += 1
        print(testCounter)

    with open("allExamples.txt", "w") as f:
        f.write(ccb_string)
    f.close()
#
while lowestPositiveReward < stop_lowest_positive_Reward:
    # while iteration < 5000:
    if iteration > 100000:
        break

    if not LEARN:
        print("break")
        break
    if LEARN:
        search = True
        while search:
            randomNr = random.randint(0, len(my_data) - 1)
            count = 0
            while count < len(my_data):
                if randomNr != count and my_data[count]["positiveRewards"] < stop_lowest_positive_Reward:
                    search = False
                count += 1

    shared_example = "ccb shared " + my_data[randomNr]["contextString"]

    actions_example = ""
    count = 0
    index_action = []
    while count < len(my_data):
        if count != randomNr:
            actions_example += "ccb action" + my_data[count]["contextString"]
            index_action.append(count)
        count += 1

    slots_pred = ""
    count = 0
    while count < NUMBER_OF_SLOTS:
        slots_pred += "ccb slot |\n"
        count += 1

    pred_example = shared_example + actions_example + slots_pred
    pred = vw.predict(pred_example)

    pred_value = []
    slot_learn = ""
    for slot_pred in pred:
        prededSlot = slot_pred[0][0]
        if my_data[index_action[prededSlot]]["clusterCat"] == my_data[randomNr]["clusterCat"]:
            cost = -1
            pred_value.append(cost * -1)
            my_data[index_action[prededSlot]]['positiveRewards'] += 1
            slot_learn += "ccb slot " + str(slot_pred[0][0]) + ":-1:" + str(slot_pred[0][1]) + " |\n"
            # print(slot_learn)
        else:
            cost = 0
            pred_value.append(cost * -1)
            slot_learn += "ccb slot " + str(slot_pred[0][0]) + ":0:" + str(slot_pred[0][1]) + " |\n"
        # print(cost)

    learn_example = shared_example + actions_example + slot_learn

    # print(learn_example)
    vw_format = vw.parse(learn_example, vowpalwabbit.LabelType.CONDITIONAL_CONTEXTUAL_BANDIT)
    vw.learn(vw_format)

    pred_value_iteration.append(pred_value)

    pred_value.sort(reverse=True)
    pred_average = 0
    mult1 = [1]
    mult2 = [0.7, 0, 3]
    mult3 = [0.5, 0, 3, 0.2]
    mult4 = [0.4, 0.3, 0.2, 0.1]
    c = 0
    for v in pred_value:
        if NUMBER_OF_SLOTS == 1:
            pred_average += v * mult1[c]
        elif NUMBER_OF_SLOTS == 2:
            pred_average += v * mult2[c]
        elif NUMBER_OF_SLOTS == 3:
            pred_average += v * mult3[c]
        elif NUMBER_OF_SLOTS == 4:
            pred_average += v * mult4[c]
        else:
            pred_average += v * 1 / NUMBER_OF_SLOTS
        c += 1

    highest_value = pred_value[0]
    highest_pred_value.append(highest_value)

    cumulutaded_highest_pred_value = 0

    for a in highest_pred_value:
        cumulutaded_highest_pred_value += a

    cumulutaded_highest_pred_value = cumulutaded_highest_pred_value / (iteration + 1)
    cumulated_average_highest_pred_value.append(cumulutaded_highest_pred_value)
    # pred_average=pred_average/len(pred_value)

    average_pred_value.append(pred_average)
    cumulated_average = 0
    for a in average_pred_value:
        cumulated_average += a
    cumulated_average = cumulated_average / (iteration + 1)
    cumulated_average_pred.append(cumulated_average)
    iterations.append(iteration + 1)

    iteration += 1
    lowestPositiveReward = 9999
    rewards_zero = 0
    tickets_not_stop_condition = 0
    for ticket in my_data:
        if ticket['positiveRewards'] < lowestPositiveReward:
            lowestPositiveReward = ticket['positiveRewards']
        if ticket['positiveRewards'] == 0:
            rewards_zero += 1
        if ticket['positiveRewards'] < stop_lowest_positive_Reward:
            tickets_not_stop_condition += 1

    if (iteration % 500 == 0):
        vw.save('checkpoints/vw_model_from_json_in_iterations_randomshuffle_' + str(iteration) + '.vow')
        pickle_files = [my_data,
                        NUMBER_OF_SLOTS,
                        eval_data,
                        iterations,
                        iteration,
                        average_pred_value,
                        highest_pred_value,
                        cumulated_average_highest_pred_value,
                        cumulated_average_pred,
                        pred_value_iteration,
                        lowestPositiveReward,
                        stop_lowest_positive_Reward,
                        actions]
        with open('checkpoints/pickle_' + str(iteration) + '.pkl', 'wb') as f:
            pickle.dump(pickle_files, f)

        print("iteration: " + str(iteration))
        print(cumulated_average_highest_pred_value[-1])
        print("r " + str(rewards_zero))
        print("l " + str(lowestPositiveReward))
        print("not stop condition: " + str(tickets_not_stop_condition))

plt.title("Cumulated Average Reward - eps")
plt.plot(iterations, cumulated_average_pred, label="Weighted Average Reward")
plt.plot(iterations, cumulated_average_highest_pred_value, label="Highest Slot Average Reward")
plt.xlabel("Iterations")
plt.ylabel("Average Cumulated Reward")
# plt.title("Cumulated Average Reward of Multi-Armed Bandit")
plt.legend()
plt.show()
if LEARN:
    vw.save('vw_model_from_json_in_iterations_randomshuffle.vow')
    vw.finish()

actions_example = ""
index_action = []
count = 0
while count < len(my_data):
    actions_example += "ccb action" + my_data[count]["contextString"]
    index_action.append(count)
    count += 1

count = 0
slots_pred = ""
while count < NUMBER_OF_SLOTS:
    slots_pred += "ccb slot |\n"
    count += 1

true_label = []
pred_label = []

cat2int = {}
int2cat = {}

cats = []
for t in my_data:
    cats.append(t["clusterCat"])

cats = list(set(cats))
cats.sort()
i = 0
for cat in cats:
    cat2int[cat] = i
    int2cat[i] = cat
    i += 1
print(cat2int)
print(int2cat)

rewards = []

for t in eval_data:
    count = 0
    shared_example = "ccb shared " + t["contextString"]
    pred_example = shared_example + actions_example + slots_pred
    pred = vw.predict(pred_example)
    highest_reward = 0
    t_label = cat2int[t["clusterCat"]]

    p_label = cat2int[my_data[index_action[pred[0][0][0]]]["clusterCat"]]
    for slot_pred in pred:
        prededSlot = slot_pred[0][0]
        # pred_label.append(cat2int[my_data[index_action[prededSlot]]["clusterCat"]])
        if my_data[index_action[prededSlot]]["clusterCat"] == t["clusterCat"]:
            p_label = cat2int[my_data[index_action[prededSlot]]["clusterCat"]]
            highest_reward += 1
    true_label.append(t_label)
    pred_label.append(p_label)

print()

with open("eval_data.json", "w", encoding="utf8") as f:
    json.dump(eval_data, f)
a = pd.read_json("eval_data.json")
a.to_excel("eval_data.xlsx")

matrix = confusion_matrix(true_label, pred_label)

newCat2Int = {}
for a in pred_label:
    newCat2Int[int2cat[a]] = a
for a in true_label:
    newCat2Int[int2cat[a]] = a

countCat = {}
print(pred_label)
print(true_label)
df_cm = pd.DataFrame(matrix, index=[i for i in newCat2Int],
                     columns=[i for i in newCat2Int])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()

ax = plt.subplot()
sn.heatmap(df_cm, fmt='g', ax=ax, xticklabels=True,
           yticklabels=True)  # annot=True, fmt='g', ax=ax, xticklabels=True, yticklabels=True);  # annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
textstr = "Accuracy " + "\n" + str(
    "%.4f" % accuracy_score(true_label, pred_label)) + "\n" + "\n" \
          + "Precision (Weighted) " + "\n" + str(
    "%.4f" % precision_score(true_label, pred_label, average='weighted')) + "\n" + "\n" \
          + "Recall(Weighted) " + "\n" + str(
    "%.4f" % precision_score(true_label, pred_label, average='weighted')) + "\n" + "\n" \
          + "F1 Score (Weighted) " + "\n" + str("%.4f" % f1_score(true_label, pred_label, average='weighted'))

ax.text(-0.45, -0.0, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top')

ax.set_title('Confusion Matrix - RndShuffle all tickets \n with kat label')
plt.tight_layout()
plt.show()

print("Wabbit Accuracy Score -> ", accuracy_score(true_label, pred_label))

print("Wabbit Precision Score (AVERAGE_NONE)-> ", precision_score(true_label, pred_label, average=None))
print("Wabbit Precision Score (AVERAGE_micro)-> ", precision_score(true_label, pred_label, average='micro'))
print("Wabbit Precision Score (AVERAGE_macro)-> ", precision_score(true_label, pred_label, average='macro'))
print("Wabbit Precision Score (AVERAGE_weighted)-> ", precision_score(true_label, pred_label, average='weighted'))

print("Wabbit Recall Score(AVERAGE_NONE) -> ", recall_score(true_label, pred_label, average=None))
print("Wabbit Recall Score(AVERAGE_micro) -> ", recall_score(true_label, pred_label, average='micro'))
print("Wabbit Recall Score(AVERAGE_macro) -> ", recall_score(true_label, pred_label, average='macro'))
print("Wabbit Recall Score(AVERAGE_weighted) -> ", recall_score(true_label, pred_label, average='weighted'))

print("Wabbit F1 Score (AVERAGE_NONE)-> ", f1_score(true_label, pred_label, average=None))
print("Wabbit F1 Score (AVERAGE_micro)-> ", f1_score(true_label, pred_label, average='micro'))
print("Wabbit F1 Score (AVERAGE_macro)-> ", f1_score(true_label, pred_label, average='macro'))
print("Wabbit F1 Score (AVERAGE_weighted)-> ", f1_score(true_label, pred_label, average='weighted'))
