#!/bin/bash

# script de preprocess de Wikipedia pour le modèle de Kolitsas à partir des entités d'embeddings appris

#############################################################################################################
## ARGS
LANG=$1
MODE=$2
DATA=$3
if [ "$LANG" == "" ]; then
    echo "LANG must be 'FR', 'EN' or 'EN19'"
    exit
fi
if [ "$MODE" == "" ] || [ "$MODE" == "ALL" ]; then
    echo "DO ALL PROCEDURE"
    MAKE_ENTITIES="DO"
    MAKE_PREPROCESS="DO"
    MAKE_TF_FILES="DO"
elif [ "$MODE" == "PREPROCESS" ]; then
    echo "ONLY PREPROCESS DATA WITH EXISTING PEM AND ENTITIES"
    MAKE_ENTITIES="DONT"
    MAKE_PREPROCESS="DO"
    MAKE_TF_FILES="DO"
elif [ "$MODE" == "WIKI2VEC" ]; then
    echo "ONLY REPROCESS PEM AND ENTITIES"
    MAKE_ENTITIES="DO"
    MAKE_PREPROCESS="DONT"
    MAKE_TF_FILES="DONT"
elif [ "$MODE" == "TF" ]; then
    echo "ONLY COMPUTE PREPRO_UTIL.PY"
    MAKE_ENTITIES="DONT"
    MAKE_PREPROCESS="DONT"
    MAKE_TF_FILES="DO"
else
    echo "MODE must be 'ALL' / 'WIKI2VEC' / 'PREPROCESS' OR 'TF'"
    exit
fi
if [ "$DATA" == "" ]; then
    echo "CHOOSE $LANG ALL DATA"
    if [ "$LANG" == "FR" ]; then
        DATA="WIKI DB TR MINI"
    else
        DATA="WIKI AIDA"
    fi
else
    echo "CHOOSE $LANG DATA : $DATA"
fi

## PATHS
CURRENT_PATH=$(pwd)
PROCESS_PATH="/people/carpentier/Modeles/all_datasets_wiki"
PICK_FOLDER="$PROCESS_PATH/nel-wikipedia${LANG}_train_1/"
DBPEDIA_TRAIN="$DB_PROCESS_PATH/DBpedia_train"
DATA_PATH="/people/carpentier/Modeles/end2end_neural_el-master/code" #"/home/carpentier/Modèles/end2end/code"
RLTD_PATH="/people/carpentier/Modeles/en_entities/deep-ed-master_${LANG}"
GLOBAL_ENT_PATH="/people/carpentier/Modeles/en_entities" #"/vol/usersiles2/carpentier/save_entities_model"
FR_TRUE_ENT="deep-ed-master_${LANG}" #"generated_fr_09_true_map"
FR_WRONG_ENT="generated_fr_08_wrong_map" #Never used

## EXPERIENCE & LOG
DATA_TYPE="nel${LANG}"
LOG_STDOUT="$CURRENT_PATH/log_process_wikidump.stdout"

## FILES NAME
EXP="Kolitsas_${LANG}"
CROSSWIKI_NAME="crosswikis_wikidump${LANG}_p_e_m.txt"
WIKI_NAME_ID_MAP="wiki_name_id_map_${LANG}.txt"
WIKI_NAME_ID_MAP_UNIFY="wiki_name_id_map_unify_${LANG}.txt"
WIKIID2NNID="wikiid2nnid_${LANG}.txt" # important to generate all the good files name in 'wiki2vec_txt_from_npy'
DATASETS="new_datasets_wiki2vec_${LANG}"
DATASETS_NAME="w2v${LANG}"
if [ "$LANG" == "FR" ]; then
    ENT_VECS="frwiki_20211020_300d.txt"
    ENTLANG="--entity_language=fr"
    WIKI_PATH="/vol/usersiles2/carpentier/save_model/textWithAnchorsFromAllWikipedia2021Oct_fr.txt"
elif [ "$LANG" == "EN" ]; then
    ENT_VECS="enwiki_20211120_300d.txt"
    ENTLANG="--entity_language=en"
    WIKI_PATH="/vol/usersiles2/carpentier/save_model/textWithAnchorsFromAllWikipedia2021Jun.txt"
elif [ "$LANG" == "EN19" ]; then
    ENT_VECS="enwiki_20211120_300d.txt" # "enwiki_20180420_300d.txt"
    ENTLANG="--entity_language=en"
    WIKI_PATH="/vol/usersiles2/carpentier/save_model/textWithAnchorsFromAllWikipedia2021Jun.txt"
else
    echo "LANG must be 'FR', 'EN' or 'EN19'"
    exit
fi

## OPTIONS
# 0 = use /deep-ed-master_true_map/.../wiki_name_id_map.txt to find id of wiki2vec entities
# 1 = generate new ids for wiki2vec entities and clean cross_wikipedia_p_e_m.txt to align with existing ids
NEW_INDEX=0 
CONVERT_TO_TR=0
LENDOC=100000 #Size of the final dataset
MINDOC=1000 #Min characters of all docs
MAXDOC=5000 #Max characters of all docs
#############################################################################################################

if [ "$MAKE_ENTITIES" == "DO" ]; then
    echo "########################################################"
    echo ">>>>>>>>>>>>>> Create $DATA_TYPE Corpus <<<<<<<<<<<<<<<<"
    echo "########################################################"
    echo "copy $LENDOC documents from initial corpus héhé"
    rm -rf $DATA_PATH/../../en_entities/$DATA_TYPE/train #delete old picking
    mkdir -p $DATA_PATH/../../en_entities/$DATA_TYPE/train
    LENDATASET=$(find "$DATA_PATH/../../en_entities/$DATA_TYPE/train/" -maxdepth 1 -type f | wc -l)
    echo "$LENDATASET/0 initial docs in dataset folder"
    cd $PROCESS_PATH
    python pick_nel_documents.py --folder="$PICK_FOLDER" --save="$DATA_PATH/../../en_entities/$DATA_TYPE/train/" --lendoc=$LENDOC --min_lenght=$MINDOC --max_lenght=$MAXDOC
    LENDATASET=$(find "$DATA_PATH/../../en_entities/$DATA_TYPE/train/" -maxdepth 1 -type f | wc -l)
    echo "$LENDATASET/$LENDOC files in final dataset folder"

    echo "#########################################################################"
    echo ">>>>>>>>>>>>>> find textWithAnchorsFromAllWikipedia file <<<<<<<<<<<<<<<<"
    echo "#########################################################################"
    cd $RLTD_PATH #move to deep-ed code folder (lua)
    rm -r $RLTD_PATH/generated
    rm $RLTD_PATH/basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt
    cp $WIKI_PATH $RLTD_PATH/basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt # "textWithAnchorsFromAllWikipedia2014Feb.txt" is the generic name for the lua code
    mkdir -p $RLTD_PATH/generated
    echo "Generated wiki_name_id_map file"
    python preprocess_wikipedia.py --folder="$RLTD_PATH/basic_data/" # create new map name id directly in basic_data from textWithAnchorsFromAllWikipedia2014Feb.txt

    # preprocess
    # extract wiki dump

    echo "####################################################"
    echo ">>>>>>>>>>>>>> Generated p_e_m file <<<<<<<<<<<<<<<<"
    echo "####################################################"
    rm $RLTD_PATH/generated/ent_name_id_map.t7 #old entity map to recreate (when 'fr' or 'en')
    th data_gen/gen_p_e_m/gen_p_e_m_from_wiki.lua -root_data_dir $RLTD_PATH/ #generated 
    python empty_size_crosswiki.py --file="$RLTD_PATH/generated/wikipedia_p_e_m.txt" #check p_e_m file
    mv "$RLTD_PATH/generated/wikipedia_p_e_m.txt" "$RLTD_PATH/generated/$CROSSWIKI_NAME" # change generic name to CROSSWIKI_NAME at this step
    echo "exemple PEM final"
    head -n 3 $RLTD_PATH/generated/$CROSSWIKI_NAME

    echo "#########################################################################"
    echo ">>>>>>>>>>>>>> preprocessing of wikipedia2vec embeddings <<<<<<<<<<<<<<<<"
    echo "#########################################################################"
    cd $DATA_PATH # move to end2end folder (python)
    # this part create and/or copy all the file for the entities
    python -m make_entities_list --in_folder="../../Wikipedia2Vec/" --data_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --replace_mode $ENTLANG #create or replace entities_list file
    if [ "$LANG" == "FR" ]; then
        python -m make_entities_list --in_folder="../../Wikipedia2Vec/" --data_folder="$DATA_PATH/../../en_entities/DBpedia/" --append_mode $ENTLANG #add DB entities to entities_list file
        python -m make_entities_list --in_folder="../../Wikipedia2Vec/" --data_folder="$DATA_PATH/../../en_entities/TR/" --append_mode $ENTLANG #add TR entities to entities_list file
    else
        echo "TODO : add 'make_entities_list' for AIDA/CoNLL"
    fi
    if [ "$NEW_INDEX" -eq 0 ]; then
        echo "use existing id"
        python -m wiki2vec_txt_from_npy --in_folder="../../Wikipedia2Vec/" --folder_wiki_base="$RLTD_PATH/basic_data/" --wikifile="$ENT_VECS" --not-generate-index --wikiid2nnid="$WIKIID2NNID" $ENTLANG
        cp "$GLOBAL_ENT_PATH/$FR_TRUE_ENT/basic_data/wiki_name_id_map.txt" "$DATA_PATH/../data/basic_data/$WIKI_NAME_ID_MAP" #copy the real wiki_name_id_map align with the p_e_m
    else
        echo "generate new index"
        python -m wiki2vec_txt_from_npy --in_folder="../../Wikipedia2Vec/" --folder_wiki_base="$RLTD_PATH/basic_data/" --wikiid2nnid="$WIKIID2NNID" $ENTLANG
    fi

    # merge crosswiki pem & yago pem
    cp "$RLTD_PATH/generated/$CROSSWIKI_NAME" "$DATA_PATH/../data/basic_data"
    mv "$DATA_PATH/../data/basic_data/crosswikis_wikipedia_p_e_m.txt" "$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" 
    if [ "$NEW_INDEX" -eq 1 ]; then # not process if index already came from wiki_name_id_map
        echo "clean cross_wiki with generate ids"
        python -m clean_crosswikis --file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --wiki_name_id_map="$WIKI_NAME_ID_MAP"
    fi
fi

if [ "$MAKE_PREPROCESS" == "DO" ]; then
    echo "########################################################"
    echo ">>>>>>>>>>>>>> verif mentions integrity <<<<<<<<<<<<<<<<"
    echo "########################################################"
    cd $DATA_PATH # move to end2end folder (python)
    # check mentions between datasets and p_e_m
    #python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
    if [ "$LANG" == "EN" ] || [ "$LANG" == "EN19" ]; then
        if grep -q "AIDA" <<< "$DATA"; then
            echo "------------ verif AIDA ------------"
            python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../data/basic_data/test_datasets/AIDA/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name $ENTLANG
        fi
        if grep -q "WIKI" <<< "$DATA"; then
            echo "------------ verif WIKI EN ------------"
            python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name $ENTLANG
        fi
    elif [ "$LANG" == "FR" ]; then
        if grep -q "WIKI" <<< "$DATA"; then
            echo "------------ verif WIKI FR ------------"
            python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name $ENTLANG
        fi
        if grep -q "DB" <<< "$DATA"; then
            echo "------------ verif DB ------------"
            python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/DBpedia/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name $ENTLANG
        fi
        if grep -q "TR" <<< "$DATA"; then
            echo "------------ verif TR ------------"
            python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/TR/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name $ENTLANG
        fi
    else
        echo "LANG must be 'FR', 'EN' or 'EN19'"
        exit
    fi
    echo "Compatibility between wikiid2nnid & wiki_name_id_map"
    python -m compatibilité_mapid_nnid --wiki_id_map_file=$WIKI_NAME_ID_MAP --wikiid2nnid_file=$WIKIID2NNID

    echo "###############################################################"
    echo ">>>>>>>>>>>>>> preprocessing of Datasets $LANG <<<<<<<<<<<<<<<<"
    echo "###############################################################"
    if [ "$LANG" == "EN" ] || [ "$LANG" == "EN19" ]; then
        # preprocess Wikipedia
        if grep -q "WIKI" <<< "$DATA"; then
            echo "------------ preprocess WIKI EN ------------"
            python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --split_to_test --unify_entity_name $ENTLANG #>> $LOG_STDOUT
            mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_train.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_dev.txt"4
        fi
        # preprocess AIDA
        if grep -q "AIDA" <<< "$DATA"; then
            echo "------------ preprocess AIDA ------------"
            python -m preprocessing.prepro_aida --aida_folder="$DATA_PATH/../data/basic_data/test_datasets/AIDA/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name
            mv "$DATA_PATH/../data/$DATASETS/aida_train.txt" "$DATA_PATH/../data/$DATASETS/aida_${DATASETS_NAME}_train.txt"
            mv "$DATA_PATH/../data/$DATASETS/aida_test.txt" "$DATA_PATH/../data/$DATASETS/aida_${DATASETS_NAME}_test.txt"
            mv "$DATA_PATH/../data/$DATASETS/aida_dev.txt" "$DATA_PATH/../data/$DATASETS/aida_${DATASETS_NAME}_dev.txt"
        fi
    elif [ "$LANG" == "FR" ]; then
        # preprocess Wikipedia
        if grep -q "WIKI" <<< "$DATA"; then
            echo "------------ preprocess WIKI FR ------------"
            python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --split_to_test --unify_entity_name #>> $LOG_STDOUT
            mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_train.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_dev.txt"
        fi
        # preprocess DBpedia
        if grep -q "DB" <<< "$DATA"; then
            echo "------------ preprocess DB ------------"
            python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/DBpedia/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --split_to_test --unify_entity_name #>> $LOG_STDOUT
            mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/DB_${DATASETS_NAME}_train.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/DB_${DATASETS_NAME}_test.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/DB_${DATASETS_NAME}_dev.txt"
        fi
        # preprocess TR FR
        if grep -q "TR" <<< "$DATA"; then
            echo "------------ preprocess TR ------------"
            python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/TR/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name #>> $LOG_STDOUT
            mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_train.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_test.txt"
            mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_dev.txt"
            python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/TR/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --hard_only --unify_entity_name #>> $LOG_STDOUT
            mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_test_hard.txt"
            rm "$DATA_PATH/../data/$DATASETS/TR_train.txt"
            rm "$DATA_PATH/../data/$DATASETS/TR_dev.txt"
        fi
        if grep -q "MINI" <<< "$DATA"; then
            echo "------------ preprocess WIKIMINI & DBMINI ------------"
            python -m preprocessing.prepro_KILT --input_path="$DATA_PATH/../../en_entities/KILT" --output_path="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP_UNIFY" --unify_entity_name --input_filenames="DBMINI_train.jsonl|DBMINI_dev.jsonl|DBMINI_test.jsonl|WIKIMINI_train.jsonl|WIKIMINI_dev.jsonl"
            mv "$DATA_PATH/../data/$DATASETS/DBMINI_train.txt" "$DATA_PATH/../data/$DATASETS/DBMINI_${DATASETS_NAME}_train.txt"
            mv "$DATA_PATH/../data/$DATASETS/DBMINI_test.txt" "$DATA_PATH/../data/$DATASETS/DBMINI_${DATASETS_NAME}_test.txt"
            mv "$DATA_PATH/../data/$DATASETS/DBMINI_dev.txt" "$DATA_PATH/../data/$DATASETS/DBMINI_${DATASETS_NAME}_dev.txt"
            mv "$DATA_PATH/../data/$DATASETS/WIKIMINI_train.txt" "$DATA_PATH/../data/$DATASETS/WIKIMINI_${DATASETS_NAME}_train.txt"
            mv "$DATA_PATH/../data/$DATASETS/WIKIMINI_dev.txt" "$DATA_PATH/../data/$DATASETS/WIKIMINI_${DATASETS_NAME}_dev.txt"            
        fi
    else
        echo "LANG must be 'FR', 'EN' or 'EN19'"
        exit
    fi
    
    # insight
    echo "#####################################################"
    echo ">>>>>>>>>>>>>> Insight - Final Verif <<<<<<<<<<<<<<<<"
    echo "#####################################################"
    ls -lah "$DATA_PATH/../data/$DATASETS"
    if [ "$LANG" == "EN" ] || [ "$LANG" == "EN19" ]; then
        if grep -q "WIKI" <<< "$DATA"; then
            echo "------------ Insight WIKI EN ------------"
            python -m insight --new_dataset="$DATASETS" --train="wiki${DATATYPE}_${DATASETS_NAME}_train.txt" --dev="wiki${DATATYPE}_${DATASETS_NAME}_dev.txt" --test="wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
        fi
        if grep -q "WIKI" <<< "$DATA"; then
            echo "------------ Insight AIDA ------------"
            python -m insight --new_dataset="$DATASETS" --train="aida_${DATASETS_NAME}_train.txt" --dev="aida_${DATASETS_NAME}_dev.txt" --test="aida_${DATASETS_NAME}_test.txt"
        fi
    elif [ "$LANG" == "FR" ]; then
        if grep -q "WIKI" <<< "$DATA"; then
            echo "------------ Insight WIKI FR ------------"
            python -m insight --new_dataset="$DATASETS" --train="wiki${DATATYPE}_${DATASETS_NAME}_train.txt" --dev="wiki${DATATYPE}_${DATASETS_NAME}_dev.txt" --test="wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
        fi
        if grep -q "DB" <<< "$DATA"; then
            echo "------------ Insight DB ------------"
            python -m insight --new_dataset="$DATASETS" --train="DB_${DATASETS_NAME}_train.txt" --dev="DB_${DATASETS_NAME}_dev.txt" --test="DB_${DATASETS_NAME}_test.txt"
        fi
        if grep -q "TR" <<< "$DATA"; then
            echo "------------ Insight TR ------------"
            python -m insight --new_dataset="$DATASETS" --train="TR_${DATASETS_NAME}_train.txt" --dev="TR_${DATASETS_NAME}_dev.txt" --test="TR_${DATASETS_NAME}_test.txt"
        fi
        if grep -q "MINI" <<< "$DATA"; then
          echo "------------ Insight MINI ------------"  
          python -m insight --new_dataset="$DATASETS" --train="DBMINI_${DATASETS_NAME}_train.txt" --dev="DBMINI_${DATASETS_NAME}_dev.txt" --test="DBMINI_${DATASETS_NAME}_test.txt"
          python -m insight --new_dataset="$DATASETS" --train="WIKIMINI_${DATASETS_NAME}_train.txt" --dev="WIKIMINI_${DATASETS_NAME}_dev.txt"
        fi
    else
        echo "LANG must be 'FR', 'EN' or 'EN19'"
        exit
    fi    
fi

if [ "$MAKE_TF_FILES" == "DO" ]; then
    # preprocess final 
    echo "###################################################"
    echo ">>>>>>>>>>>>>> Final Preprocessing <<<<<<<<<<<<<<<<"
    echo "###################################################"
    cd $DATA_PATH # move to end2end folder (python)
    python -m preprocessing.prepro_util --p_e_m_choice="crosswiki" --experiment_name="$EXP" --persons_coreference=False --persons_coreference_merge=False --wiki_id_file="$WIKI_NAME_ID_MAP" --wikiid2nnid_file="$WIKIID2NNID" --prob_p_e_m="$CROSSWIKI_NAME" --datasets_folder="$DATASETS" --vocab_file="vocab_freq_${LANG}.pickle" #>> $LOG_STDOUT #--lowercase_spans=True --lowercase_p_e_m=True
fi

echo "########################################"
echo ">>>>>>>>>>>>>> ALL DONE <<<<<<<<<<<<<<<<"
echo "########################################"
