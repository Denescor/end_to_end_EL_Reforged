#!/bin/bash

################################################################################
# General Parameters
GENERAL_F="/people/carpentier/Modeles/all_datasets_wiki" # General folder
LANG="fr" #lang of the wikipedia dump
DUMP="20211020" #"20211120" #date of the wikipedia dump
DATE_DUMP="2021Oct" #dat in letter for the final txt file

# Variables
WIKI_F="$GENERAL_F/${LANG}_${DUMP}_extracted"
ORIGINAL_DUMP="$GENERAL_F/dump_zip/${LANG}wiki-${DUMP}-pages-articles-multistream.xml.bz2"
DUMP_ZIP="$GENERAL_F/${LANG}wiki-${DUMP}-pages-articles-multistream.xml.bz2"
DUMP_REDIRECT="$GENERAL_F/${LANG}wiki-${DUMP}-pages-articles-multistream.xml.bz2"
DUMP_UNZIP="$GENERAL_F/${LANG}wiki-${DUMP}-pages-articles-multistream.xml"
TXT_NAME="$GENERAL_F/textWithAnchorsFromAllWikipedia${DATE_DUMP}.txt"
TXT_BRUT="$GENERAL_F/textWithAnchorsFromAllWikipedia${DATE_DUMP}_brut.txt"
PYTHON_F="$GENERAL_F"

# Options
# 0 : do not process redirect for the wiki dump
# 1 : process redirect for the wiki dump
REDIRECT=$1
if [ -z "$REDIRECT" ]; then REDIRECT=0 #no arg = no redirect
elif [ "$REDIRECT" = "1" ]; then REDIRECT=1 #convert string to int
else REDIRECT=0 #convert string to int
fi
WIKIEXTRACTOR=4 #wikiextractor version

################################################################################

cd $PYTHON_F
echo "$(pwd)"

if [ -a "$DUMP_ZIP" ]; then
	echo "files already copied"
else
	cp "$GENERAL_F/dump_zip/${LANG}wiki-${DUMP}-pages-articles-multistream.xml.bz2" "$GENERAL_F"
	cp "$GENERAL_F/dump_zip/${LANG}wiki-${DUMP}-pages-articles.xml.bz2" "$GENERAL_F"
	echo "files copied"
fi

if [ "$REDIRECT" = 1 ]; then
        echo "process redirect for $DUMP_REDIRECT"
        bzip2 -d "$DUMP_REDIRECT"
        java -cp $PYTHON_F/wikipedia_redirect/codebase/bin edu.cmu.lti.wikipedia_redirect.WikipediaRedirectExtractor "$DUMP_UNZIP"
        rm "$DUMP_UNZIP"
	mv "$PYTHON_F/target/wikipedia_redirect.txt" "$PYTHON_F/target/${LANG}wikipedia_redirect-${DUMP}.txt"
	mv "$PYTHON_F/target/wikipedia_redirect.ser" "$PYTHON_F/target/${LANG}wikipedia_redirect-${DUMP}.ser"
else #no redirect
        echo "no redirect require."
fi
echo "REDIRECT DONE"
echo "extract dump with wikiextractor"
if [ "$WIKIEXTRACTOR" = 3 ]; then
	python $PYTHON_F/wikiextractor/WikiExtractor3.py $DUMP_ZIP -o $WIKI_F -l
elif [ "$WIKIEXTRACTOR" = 2 ]; then
	python3 $PYTHON_F/wikiextractor/WikiExtractor.py $DUMP_ZIP -o $WIKI_F -l
else
	echo "ERROR : no version $WIKIEXTRACTOR found for wikiextracor"
fi
echo "generated $TXT_BRUT with the extracted dump"
python $PYTHON_F/read_wikiprepro.py --wiki_extract="$WIKI_F" --output_file="$TXT_BRUT" --print_output
echo "generated $TXT_NAME from brut file"
python $PYTHON_F/from_url_to_utf8.py --input_file=$TXT_BRUT --output_file=$TXT_NAME
echo "WIKIEXTRACTOR DONE"
echo "ALL DONE"
ls -lah $GENERAL_F

