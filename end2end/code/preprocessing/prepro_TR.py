import os
import tqdm
import random
import argparse
import preprocessing.util as util
from wiki2vec_txt_from_npy import unify_entity_name 

def create_lowerdict(wiki_name_id_map):
    dico_id = dict()
    dico_lower = dict()
    dico_temp, _ = util.load_wiki_name_id_map(filepath=wiki_name_id_map, verbose=False)
    for name, i in dico_temp.items(): dico_id[unify_entity_name(name)] = i
    for name in dico_id.keys(): 
        try:
            lowername = name.lower()
            assert lowername.islower(), "erreur dans la mise en lowercase"
            dico_lower[lowername] = name
        except AssertionError : continue
    print("taille wiki\ncasual : {}\nlowercase : {}".format(len(dico_id), len(dico_lower)))
    print("{}".format(list(dico_id.keys())[:15]))
    assert len(dico_lower)>0, "erreur dans la mise en lowercase"
    return dico_id, dico_lower

def process_TR_file(txt_filepath, mention_filepath, fout, entityNameIdMap, dico_lower):
    unknown_gt_ids = 0   # counter of ground truth entity ids that are not in the wiki_name_id.txt
    nb_mentions = 0
    nb_n = 0
    nb_not_lower = 0
    nb_lines = 0
    with open(mention_filepath) as fin:
        mention_dict = dict()
        for line in fin:
            try:
                nb_lines += 1
                bg, nd, ent_en, ent_fr, hard = line.split("\t")
                if args.entity_language == "fr": 
                    if args.unify: final_ent = unify_entity_name(ent_fr) #marche car il n'y peut pas y avoir 2 mentions dans 1 documents qui débutent au même endroit
                    else: final_ent = ent_fr
                else: 
                    try: 
                        if ent_en.islower(): ent_en = dico_lower[unify_entity_name(ent_en)] # On repasse les entités lower case en casual case si ce n'est pas le cas
                    except KeyError :# ça ne marche pas == l'entité n'est pas dans connue dans AIDA (ou autre erreur qui rendra l'entité introuvable)
                        nb_not_lower += 1
                        continue # On ne trouve pas l'entité, on l'exclue et on passe à la suite
                    if args.unify: final_ent = unify_entity_name(ent_en)
                    else: final_ent = ent_en
                mention_dict[int(bg)] = (int(nd), final_ent, hard)
                nb_mentions += 1
            except ValueError: print(">> ValueError at line '{}' of file '{}'".format(line,mention_filepath))
    if nb_mentions == 0: return unknown_gt_ids, nb_lines, nb_n, nb_not_lower # On ne process pas de documents sans mentions
    with open(txt_filepath) as fin :
        text = ""
        text_anno = ""
        current_end = -1
        process_mention = False
        for line in fin:
            text += line
        for i, car in enumerate(text): #on va placer les entités au fur et à mesure en comptant par caractère    
            if i in mention_dict and (not process_mention):
                current_mention = mention_dict[i]
                true_mention = current_mention[1]
                is_hard = bool(current_mention[2])
                ent_id = entityNameIdMap.compatible_ent_id(name=true_mention)
                if ent_id is not None:
                    if (not args.hard_only) or (args.hard_only and is_hard): 
                        text_anno += " MMSTART_{} ".format(ent_id)
                        current_end = current_mention[0]
                        process_mention = True
                else:
                    unknown_gt_ids += 1
                    #print("unknow gt ids : {} -> {} ({} - {})".format(current_mention[1],true_mention,i,current_mention[0]))
                if car == '\n': nb_n += 1
                else: text_anno += car
            elif i == current_end - 1 and (process_mention):
                if car == '\n': 
                    text_anno += "MMEND "
                    nb_n += 1
                else: text_anno += "{} MMEND ".format(car)
                process_mention = False
            elif car != '\n':
                text_anno += car
            else: nb_n += 1 #do not copy '\n'
        text_token = text_anno.replace('\n', '')
        text_token = [x.strip() for x in text_token.split(" ")]
        for word in text_token:
            word_split = word.split('\n')
            if word != '':
                fout.write("{}\n".format(word_split[0]))
    return unknown_gt_ids, nb_mentions, nb_n, nb_not_lower

def process_TR(folder, out_filepath):
    # _, wiki_id_name_map = util.load_wiki_name_id_map(lowercase=False)
    #_, wiki_id_name_map = util.entity_name_id_map_from_dump()
    entityNameIdMap = util.EntityNameIdMap()
    entityNameIdMap.init_compatible_ent_id(wiki_map_file=args.wiki_path)
    print(list(entityNameIdMap.wiki_name_id_map.keys())[:15])
    _, dico_lower = create_lowerdict(args.wiki_path)
    unknown_gt_ids = 0   # counter of ground truth entity ids that are not in the wiki_name_id.txt
    nb_not_lower = 0
    nb_mention = 0
    nb_n = 0
    nb_process = 0
    nb_lines = 0
    os.chdir(folder)
    with open(out_filepath, 'w') as fout:
        list_doc = [os.path.splitext(x)[0] for x in os.listdir() if os.path.isfile(x) and os.path.splitext(x)[1]==".mentions"]
        for doc in tqdm.tqdm(list_doc, total=len(list_doc), desc="Preprocessing of {}".format(os.path.basename(os.path.normpath(out_filepath)))):
            mention_file = "{}.mentions".format(doc)
            txt_file = "{}.txt".format(doc)
            fout.write("DOCSTART_"+doc.replace(' ', '_')+"\n")
            unknown_gt_ids_temp, nb_mention_temp, nb_n_temp, nb_not_l_temp = process_TR_file(txt_file, mention_file, fout, entityNameIdMap, dico_lower)
            ## Nombre de mentions passées et compatabilisées
            if (args.entity_language == "fr") or (args.entity_language == "en" and nb_mention_temp == nb_not_l_temp) : 
                nb_process += 1
                nb_lines += nb_mention_temp # Si aucune mention n'est passé, le nombre total est le nombre initial
            else: nb_lines += nb_mention_temp + nb_not_l_temp # Sinon c'est le nombre de mentions compatabilisées + le nombre de mentions passées
            nb_not_lower += nb_not_l_temp
            ## Nombre de mentions trouvées et inconnues
            nb_mention += nb_mention_temp
            unknown_gt_ids += unknown_gt_ids_temp #par rapport au nombre total de mentions passées = nb_mention
            ## Nombre de "\n" retirées
            nb_n += nb_n_temp
            fout.write("DOCEND\n")
    print("process_TR\n\tdocuments processés : {}/{} ({:.2f}%)".format(nb_process, len(list_doc), 100*(nb_process/len(list_doc))))
    if args.entity_language == "en": print("\tmentions non repassé en casual : {}/{} ({:.2f}%)".format(nb_not_lower, nb_lines, 100*(nb_not_lower/nb_lines)))
    print("\tunknown_gt_ids: {}/{} ({:.2f}%)\n\tnb '\\n' = {}".format(unknown_gt_ids,nb_mention,100*(unknown_gt_ids/nb_mention), nb_n))     
    print("file save at '{}'".format(out_filepath))

def split_dataset(folder, old_dataset="train", new_dataset="dev", prop=0.1):
    first_folder = "{}{}/".format(folder,old_dataset)
    second_folder = "{}{}/".format(folder,new_dataset)
    if not os.path.exists(second_folder):
        os.makedirs(second_folder)
    os.chdir(first_folder)
    list_doc = [os.path.splitext(x)[0] for x in os.listdir() if os.path.isfile(x) and os.path.splitext(x)[1]==".mentions"]
    list_doc.sort() #for reproductibility
    random.seed(1158) #for reproductibility
    len_doc = int(prop*(len(list_doc))) #nombre de documents à déplacer
    pick_doc = random.sample([i for i in range(len(list_doc))], len_doc) #liste des documents à déplacer
    for i in pick_doc: #On déplace uniformément 10% des documents dans le dev
        mention_file = "{}.mentions".format(list_doc[i])
        txt_file = "{}.txt".format(list_doc[i])
        if (not os.path.isfile(first_folder+mention_file)) or (not os.path.isfile(first_folder+txt_file)):
            print("error file '{}'".format(list_doc[i]))
            continue 
        mention_file = "{}.mentions".format(list_doc[i])
        txt_file = "{}.txt".format(list_doc[i])
        os.rename(first_folder+mention_file,second_folder+mention_file)
        os.rename(first_folder+txt_file,second_folder+txt_file)
    #vérification
    os.chdir(second_folder)
    second_files = len([x for x in os.listdir() if os.path.isfile(x)])
    os.chdir(first_folder)
    first_files = len([x for x in os.listdir() if os.path.isfile(x)])
    total_files = 2*len(list_doc)
    print("{} files : {}/{} ({:.2f}%)\n{} file : {}/{} ({:.2f}%)\n".format(new_dataset, second_files, total_files, 100*(second_files/total_files), old_dataset, first_files, total_files, 100*(first_files/total_files)))
    
def remerge_dataset(folder, old_dataset="train", new_dataset="dev"):
    first_folder = "{}{}/".format(folder,old_dataset)
    second_folder = "{}{}/".format(folder,new_dataset)
    os.chdir(second_folder)
    list_second = [x for x in os.listdir() if os.path.isfile(x)]
    for doc in list_second:
        os.rename(second_folder+doc,first_folder+doc)
    #vérification
    os.chdir(first_folder)
    first_files = len([x for x in os.listdir() if os.path.isfile(x)])
    second_files = len(list_second)
    print("remerge {} & {} :\n\told {} : {} docs\n\tnew {} : {} docs".format(new_dataset,old_dataset,new_dataset,second_files,old_dataset,first_files))

def create_necessary_folders():
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--TR_folder", default="../data/basic_data/test_datasets/AIDA/")
    parser.add_argument("--hard_only", dest="hard_only", action="store_true", help="process only the hard mentions")
    parser.add_argument("--output_folder", default="../data/new_datasets/")
    parser.add_argument("--entity_language", default="fr", help="'fr' or 'en'")
    parser.add_argument("--wiki_path", default="wiki_name_id_map.txt")
    parser.add_argument("--unify_entity_name", dest="unify", action='store_true')
    parser.add_argument("--split_to_test", dest="split_test", action='store_true')
    parser.add_argument("--do_not_split", dest="split", action='store_false')
    parser.add_argument("--do_not_merge", dest="merge", action='store_false')
    parser.set_defaults(split_test=False)
    parser.set_defaults(unify=False)
    parser.set_defaults(split=True)
    parser.set_defaults(merge=True)
    parser.set_defaults(hard_only=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    create_necessary_folders()
    
    print("START {}".format(args.entity_language))
    current_dir = os.getcwd()
    print(current_dir)
    
    ## split datasets to create train / dev / test
    if args.split_test: 
        print("create test datasets from train dataset")
        split_dataset(args.TR_folder, old_dataset="train", new_dataset="test", prop=0.1) #test = 10% du train
        os.chdir(current_dir)
    if args.split: 
        print("create dev datasets from train dataset")
        split_dataset(args.TR_folder, old_dataset="train", new_dataset="dev", prop=0.15) #dev = 15% du train après création du test
        os.chdir(current_dir)
    ################################################
    
    ## preprocess datasets
    os.chdir(current_dir)
    process_TR(args.TR_folder+"train", args.output_folder+"TR_train.txt")
    os.chdir(current_dir)
    process_TR(args.TR_folder+"dev", args.output_folder+"TR_dev.txt")
    os.chdir(current_dir)
    process_TR(args.TR_folder+"test", args.output_folder+"TR_test.txt")
    os.chdir(current_dir)
    #######################
    
    ## remerge datasets to keep corpus integrity
    if args.merge: 
        remerge_dataset(args.TR_folder, old_dataset="train", new_dataset="dev") #on refusionne les deux dossiers séparés pour conserver l'intégrité du dataset initial
        os.chdir(current_dir)
    if args.split_test and args.merge: 
        remerge_dataset(args.TR_folder, old_dataset="train", new_dataset="test")
        os.chdir(current_dir)
    ############################################
    
    print("DONE")
