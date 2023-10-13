import csv

def read_text(data_path):
    f = open(data_path, 'r')
    rdr = csv.reader(f)

    labels, texts = [], []

    for i, line in enumerate(rdr):
        if i == 0:
            header = line
            title_idx = header.index('title')
            topic_idx_idx = header.index('topic_idx')
        else:
            title = line[title_idx]
            topic_idx = line[topic_idx_idx]
            labels += [topic_idx]
            texts += [title]
            

    # print('labels : ', labels)
    # print('texts : ', texts)
    return labels, texts