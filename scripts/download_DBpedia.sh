#!/bin/bash

DOWNLOAD=$1
LANG=$2

CURRENT_FOLDER=$(pwd)
DOWNLOAD_FOLDER="${CURRENT_FOLDER}/../Modeles/all_datasets_wiki/DBpedia/${LANG}"
DOWNLOAD_FOLDER2="${CURRENT_FOLDER}/../Modeles/all_datasets_wiki/DBpedia_long/${LANG}"
DBPEDIA_TRAIN="${CURRENT_FOLDER}/../Modeles/all_datasets_wiki/DBpedia_train/${LANG}"
PYTHON_F="${CURRENT_FOLDER}/../Modeles/all_datasets_wiki"

if [ "$LANG" == "fr" ]; then
	END=37
elif [ "$LANG" == "en" ]; then
	END=114
else
	echo "lang unknown"
fi
echo "$LANG has $END ttl files"

mkdir -p $DOWNLOAD_FOLDER
mkdir -p $DOWNLOAD_FOLDER2
mkdir -p $DBPEDIA_TRAIN

if [ "$DOWNLOAD" = "1" ]; then
	for i in $(seq 1 $END) 
	do
		wget -P $DOWNLOAD_FOLDER "http://downloads.dbpedia.org/2015-04/ext/nlp/abstracts/${LANG}/abstracts_${LANG}${i}.ttl.gz"
		gzip -l "${DOWNLOAD_FOLDER}/abstracts_${LANG}${i}.ttl.gz"
		gzip -d "${DOWNLOAD_FOLDER}/abstracts_${LANG}${i}.ttl.gz"
	done
	wget -P $DOWNLOAD_FOLDER2 "https://databus.dbpedia.org/dbpedia/text/long-abstracts/2021.11.01/long-abstracts_lang=${LANG}.ttl.bz2"
	gzip -d "${DOWNLOAD_FOLDER2}/long-abstracts_lang=${LANG}.ttl.bz2"
fi

cd $PYTHON_F
python ttl_to_json.py --folder_ttl="$DOWNLOAD_FOLDER"
python DBpedia_to_TR.py --folder_json="$DOWNLOAD_FOLDER" --folder_dataset="$DBPEDIA_TRAIN" --unit_test #--make_all_process


