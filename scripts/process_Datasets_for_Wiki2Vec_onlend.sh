#!/bin/bash

# script de preprocess de Wikipedia pour le modèle de Kolitsas à partir des entités d'embeddings appris

#############################################################################################################
## ARGS
LANG=$1
if [ "$LANG" == "" ]; then
    echo "LANG must be 'FR', 'EN' or 'EN19'"
    exit
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

cd $DATA_PATH # move to end2end folder (python)

echo "########################################################"
echo ">>>>>>>>>>>>>> verif mentions integrity <<<<<<<<<<<<<<<<"
echo "########################################################"
# check mentions between datasets and p_e_m
python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
if [ "$LANG" == "EN" ] || [ "$LANG" == "EN19" ]; then
    echo "------------ verif WIKI EN ------------"
    python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../data/basic_data/test_datasets/AIDA/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
    echo "------------ verif AIDA ------------"
    python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
elif [ "$LANG" == "FR" ]; then
    echo "------------ verif WIKI FR ------------"
    python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
    echo "------------ verif DB ------------"
    python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/DBpedia/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
    echo "------------ verif TR ------------"
    python -m count_mentions --p_e_m_file="$DATA_PATH/../data/basic_data/$CROSSWIKI_NAME" --TR_folder="$DATA_PATH/../../en_entities/TR/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
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
    echo "------------ preprocess WIKI EN ------------"
    python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP" --split_to_test --unify_entity_name $ENTLANG #>> $LOG_STDOUT
    #TODO rajouter version qui tokenise avec la version java de standford (AVEC nltk MAIS PAS OBLIGÉ) #>> $LOG_STDOUT
    mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_train.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_dev.txt"
    # preprocess AIDA
    echo "------------ preprocess AIDA ------------"
    python -m preprocessing.prepro_aida --aida_folder="$DATA_PATH/../data/basic_data/test_datasets/AIDA/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name
    mv "$DATA_PATH/../data/$DATASETS/aida_train.txt" "$DATA_PATH/../data/$DATASETS/aida_${DATASETS_NAME}_train.txt"
    mv "$DATA_PATH/../data/$DATASETS/aida_test.txt" "$DATA_PATH/../data/$DATASETS/aida_${DATASETS_NAME}_test.txt"
    mv "$DATA_PATH/../data/$DATASETS/aida_dev.txt" "$DATA_PATH/../data/$DATASETS/aida_${DATASETS_NAME}_dev.txt"
    # insight
    echo "#####################################################"
    echo ">>>>>>>>>>>>>> Insight - Final Verif <<<<<<<<<<<<<<<<"
    echo "#####################################################"
    ls -lah "$DATA_PATH/../data/$DATASETS"
    echo "------------ Insight WIKI EN ------------"
    python -m insight --new_dataset="$DATASETS" --train="wiki${DATATYPE}_${DATASETS_NAME}_train.txt" --dev="wiki${DATATYPE}_${DATASETS_NAME}_dev.txt" --test="wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
    echo "------------ Insight AIDA ------------"
    python -m insight --new_dataset="$DATASETS" --train="aida_${DATASETS_NAME}_train.txt" --dev="aida_${DATASETS_NAME}_dev.txt" --test="aida_${DATASETS_NAME}_test.txt"
elif [ "$LANG" == "FR" ]; then
    # preprocess Wikipedia
    echo "------------ preprocess WIKI FR ------------"
    python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/$DATA_TYPE/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP" --split_to_test --unify_entity_name #>> $LOG_STDOUT
    #TODO rajouter version qui tokenise avec la version java de standford (AVEC nltk MAIS PAS OBLIGÉ) #>> $LOG_STDOUT
    mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_train.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/wiki${DATATYPE}_${DATASETS_NAME}_dev.txt"
    # preprocess DBpedia
    echo "------------ preprocess DB ------------"
    python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/DBpedia/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP" --split_to_test --unify_entity_name #>> $LOG_STDOUT
    mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/DB_${DATASETS_NAME}_train.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/DB_${DATASETS_NAME}_test.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/DB_${DATASETS_NAME}_dev.txt"
    # preprocess TR FR
    echo "------------ preprocess TR ------------"
    python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/TR/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name #>> $LOG_STDOUT
    mv "$DATA_PATH/../data/$DATASETS/TR_train.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_train.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_test.txt"
    mv "$DATA_PATH/../data/$DATASETS/TR_dev.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_dev.txt"
    python -m preprocessing.prepro_TR --TR_folder="$DATA_PATH/../../en_entities/TR/" --output_folder="$DATA_PATH/../data/$DATASETS/" --wiki_path="$WIKI_NAME_ID_MAP" --hard_only --unify_entity_name #>> $LOG_STDOUT
    mv "$DATA_PATH/../data/$DATASETS/TR_test.txt" "$DATA_PATH/../data/$DATASETS/TR_${DATASETS_NAME}_test_hard.txt"
    rm "$DATA_PATH/../data/$DATASETS/TR_train.txt"
    rm "$DATA_PATH/../data/$DATASETS/TR_dev.txt"
    # insight
    echo "#####################################################"
    echo ">>>>>>>>>>>>>> Insight - Final Verif <<<<<<<<<<<<<<<<"
    echo "#####################################################"
    ls -lah "$DATA_PATH/../data/$DATASETS"
    echo "------------ Insight WIKI FR ------------"
    python -m insight --new_dataset="$DATASETS" --train="wiki${DATATYPE}_${DATASETS_NAME}_train.txt" --dev="wiki${DATATYPE}_${DATASETS_NAME}_dev.txt" --test="wiki${DATATYPE}_${DATASETS_NAME}_test.txt"
    echo "------------ preprocess DB ------------"
    python -m insight --new_dataset="$DATASETS" --train="DB_${DATASETS_NAME}_train.txt" --dev="DB_${DATASETS_NAME}_dev.txt" --test="DB_${DATASETS_NAME}_test.txt"
    echo "------------ preprocess TR ------------"
    python -m insight --new_dataset="$DATASETS" --train="TR_${DATASETS_NAME}_train.txt" --dev="TR_${DATASETS_NAME}_dev.txt" --test="TR_${DATASETS_NAME}_test.txt"
else
    echo "LANG must be 'FR', 'EN' or 'EN19'"
    exit
fi

# preprocess final 
date
echo "###################################################"
echo ">>>>>>>>>>>>>> Final Preprocessing <<<<<<<<<<<<<<<<"
echo "###################################################"
python -m preprocessing.prepro_util --p_e_m_choice="crosswiki" --experiment_name="$EXP" --persons_coreference=False --persons_coreference_merge=False --wiki_id_file="$WIKI_NAME_ID_MAP" --wikiid2nnid_file="$WIKIID2NNID" --prob_p_e_m="$CROSSWIKI_NAME" --datasets_folder="$DATASETS" --vocab_file="vocab_freq_${LANG}.pickle" #>> $LOG_STDOUT #--lowercase_spans=True --lowercase_p_e_m=True

#echo "build documents"
#python -m model.args_generator --experiment_name="$EXP" --training_name="documents_${LANG}"
#python -m model.yield_documents --experiment_name="$EXP" --training_name="documents_${LANG}" --no_print_predictions --el_datasets=wikiEN_w2v_dev_z_aida_w2v_dev_z_aida_w2v_test --ed_datasets=  --el_val_datasets=0 --ed_val_datasets=0 --all_spans_training=True --wikiid2nnid_name="$WIKIID2NNID"

echo "########################################"
echo ">>>>>>>>>>>>>> ALL DONE <<<<<<<<<<<<<<<<"
echo "########################################"
date
