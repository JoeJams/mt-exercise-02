#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

# mkdir -p $data/wikitext-2

# for corpus in train valid test; do
#     absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
#     ln -snf $absolute_path $data/wikitext-2/$corpus.txt
# done

# download a different interesting data set!

mkdir -p $data/wow

mkdir -p $data/wow/raw
mkdir -p $data/wow/processed
mkdir -p $data/wow/splits

# I did manual download of wow from https://huggingface.co/datasets/RUCAIBox/Open-Dialogue/blob/main/wow.tgz
# Then I only used their test set as the whole thing would have been way too big but the test set had a suitable size
# I did not manage to do this automatically with wget due to some issues with the huggingface download via xet and think the manual download is easier at this point :(
# The tgz file should be placed in the this directory and is then automatically unpacked and the relevant file is moved to the right place

tar -xzf wow.tgz -C $data/wow/raw --strip-components=1

mv $data/wow/raw/test.tgt $data/wow/raw/dia.txt # choose this file due to the suitable size (around 15k lines) and rename it to dia.txt
# delete all the files except dia.txt in raw (because we don't need them)
find $data/wow/raw -type f ! -name 'dia.txt' -delete

# preprocess slightly

cat $data/wow/raw/dia.txt | python $base/scripts/preprocess_raw.py > $data/wow/processed/dia.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/wow/processed/dia.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/wow/processed/dia.preprocessed.txt

# split into train, valid and test (I used 10000 lines in total with a 80-10-10 split)

head -n 1000 $data/wow/processed/dia.preprocessed.txt | tail -n 1000 > $data/wow/splits/valid.txt
head -n 1000 $data/wow/processed/dia.preprocessed.txt | tail -n 1000 > $data/wow/splits/test.txt
tail -n 8000 $data/wow/processed/dia.preprocessed.txt | head -n 8000 > $data/wow/splits/train.txt


# Replace the data.py and main.py in tools with the one from the repos
cp $base/scripts/data.py $tools/pytorch-examples/word_language_model/data.py
cp $base/scripts/main.py $tools/pytorch-examples/word_language_model/main.py