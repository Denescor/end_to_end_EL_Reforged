
import argparse
import model.config as config


def process_file(new_datasets,filename):
    entities = set()
    mentions = set()
    with open(config.base_folder+"data/"+new_datasets+"/"+filename) as fin:
        inmention = False
        mention_acc = []
        for line in fin:
            line = line.rstrip()     # omit the '\n' character
            if line.startswith('MMSTART_'):
                ent_id = line[8:]   # assert that ent_id in wiki_name_id_map
                entities.add(ent_id)
                inmention = True
                mention_acc = []
            elif line == 'MMEND':
                inmention = False
                mentions.add(' '.join(mention_acc))
            elif inmention:
                mention_acc.append(line)
    return entities, mentions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="aida_train.txt", help="name of the train dataset in AIDA format")
    parser.add_argument("--dev", default="aida_dev.txt", help="name of the dev dataset in AIDA format")
    parser.add_argument("--test", default="aida_test.txt", help="name of the test dataset in AIDA format")
    parser.add_argument("--new_datasets", default="new_datasets", help="name of the folder where pick the datasets files")
    args = parser.parse_args()
    
    train_entities, train_mentions = process_file(args.new_datasets, args.train)
    dev_entities, dev_mentions = process_file(args.new_datasets, args.dev)
    test_entities, test_mentions = process_file(args.new_datasets, args.test)

    print("Datasets :\n\t train : {}/{}\n\t dev : {}/{}\n\t test : {}/{}".format(args.new_datasets, args.train, args.new_datasets, args.dev, args.new_datasets, args.test))

    print(10*"#"+" Stats Entities "+10*"#")
    print("Number of entities in train = {}".format(len(train_entities)))
    print("Number of entities in dev = {}".format(len(dev_entities)))
    print("Number of entities in test = {}".format(len(test_entities)))


    inter_train_dev = len(train_entities.intersection(dev_entities))
    inter_train_test = len(train_entities.intersection(test_entities))
    inter_test_dev = len(dev_entities.intersection(test_entities))
    print("Common entities between train & dev = {} ({:.2f}% of train)".format(inter_train_dev, 100*(inter_train_dev/len(train_entities))))
    print("Common entities between train & test = {} ({:.2f}% of train)".format(inter_train_test, 100*(inter_train_test/len(train_entities))))
    print("Common entities between test & dev = {} ({:.2f}% of dev)".format(inter_test_dev, 100*(inter_test_dev/len(dev_entities))))

    print(10*"#"+" Stats Mentions "+10*"#")
    print("Number of mentions in train = {}".format(len(train_mentions)))
    print("Number of mentions in dev = {}".format(len(dev_mentions)))
    print("Number of mentions in test = {}".format(len(test_mentions)))

    inter_train_dev = len(train_mentions.intersection(dev_mentions))
    inter_train_test = len(train_mentions.intersection(test_mentions))
    inter_test_dev = len(dev_mentions.intersection(test_mentions))
    print("Common mentions between train & dev = {} ({:.2f}% of train)".format(inter_train_dev, 100*(inter_train_dev/len(train_mentions))))
    print("Common mentions between train & test = {} ({:.2f}% of train)".format(inter_train_test, 100*(inter_train_test/len(train_mentions))))
    print("Common mentions between test & dev = {} ({:.2f}% of dev)".format(inter_test_dev, 100*(inter_test_dev/len(dev_mentions))))
    
    print(30*"#")




























