#!/bin/bash

CD=$(pwd)
WIKI2VEC="$CD/../../Wikipedia2Vec"
BASIC_DATA="$CD/../data/basic_data"
EXP="transfert_en_wiki2vec"

echo "list paths:"
echo "cd = $CD"
echo "wiki2vec = $WIKI2VEC"
echo "basic data = $BASIC_DATA"
echo "------------------------------"

cp "$CD/../data/basic_data/wiki_name_id_map_EN.txt" "$WIKI2VEC"
mv "$WIKI2VEC/wiki_name_id_map_EN.txt" "$WIKI2VEC/TR_list_entities.txt"

echo "wiki2vec from txt to npy"
python -m wiki2vec_txt_from_npy --in_folder="$WIKI2VEC/" --wikifile="enwiki_20180420_300d.txt" --entfile="ent_en_wiki2vec.txt" --TR_folder="" --entity_vectors="ent_vecs_wiki2vecEN.txt" --wikiid2nnid="wikiid2nnid_wiki2vecEN.txt"

cp "$BASIC_DATA/prob_yago_crosswikis_wikipedia_p_e_m.txt" "$BASIC_DATA/prob_yago_crosswikis_wikipedia2vec_p_e_m.txt"

echo "clean prob_yago_crosswikis_wikipedia_p_e_m for wiki2vec"
python -m clean_crosswikis.py --file="$BASIC_DATA/prob_yago_crosswikis_wikipedia2vec_p_e_m.txt" --wiki_name_id_map="wiki_name_id_map_wiki2vecEN.txt"

echo "preprocessing datasets"
python -m preprocessing.prepro_aida --output_folder="../data/new_datasets_wiki2vec_aida/" --wiki_path="wiki_name_id_map_wiki2vecEN.txt" --unify_entity_name
python -m preprocessing.prepro_other_datasets  --output_folder="../data/new_datasets_wiki2vec_aida/"
python -m preprocessing.aida_insight

echo "final preprocessing"
python -m preprocessing.prepro_util --experiment_name="$EXP" --persons_coreference=False --persons_coreference_merge=False --wiki_id_file="wiki_name_id_map_wiki2vecEN.txt" --wikiid2nnid_file="wikiid2nnid_wiki2vecEN.txt" --prob_p_e_m="prob_yago_crosswikis_wikipedia2vec_p_e_m.txt" --datasets_folder="new_datasets_wiki2vec_aida"
