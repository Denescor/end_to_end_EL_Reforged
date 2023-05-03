#!/bin/bash

LANG=$1

echo "process for $LANG"
date

echo "STEP 1 : nel to DB"
#./process_wikipedia_to_TR.sh "Yes" "No" $LANG
echo "1 STEP DONE"
date

echo "STEP 2 : DB to tfrecord"
./process_WikipediaEN_for_Wiki2Vec.sh $LANG
echo "2 STEP DONE"
date
