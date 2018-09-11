SRC_LANG=bg; # source language
TGT_LANG=en; # target language

DATA_ROOT=./data/;

VAL_SPLIT=0-5000 # validation data. Note that this is not used for any model selection
TRN_SPLIT=0-5000
TRAIN_MAX_SIZE=10000 # the top N words included in training
TRANS_MAX_SIZE=200000 # the top M words include for testing

# export CUDA_VISIBLE_DEVICES=3;

# train the word embedding
python src/runner.py \
    --config_path src/config/config.json \
    --src "$SRC_LANG" --tgt "$TGT_LANG" \
    --src_vec "$DATA_ROOT"/wiki."$SRC_LANG".vec \
    --tgt_vec "$DATA_ROOT"/wiki."$TGT_LANG".vec \
    --train_max_size "$TRAIN_MAX_SIZE" \
    --save ./exp/ \
    --train 1 \
    --F_validation "$DATA_ROOT"/crosslingual/dictionaries/"$SRC_LANG"-"$TGT_LANG"."$VAL_SPLIT".txt;

# evaluate the trained embeddings
python src/eval/eval_translation.py \
    exp/bg-en/src.emb.txt exp/bg-en/tgt.trans.emb.txt \
    -d data/crosslingual/dictionaries/bg-en.5000-6500.txt \
    --cuda;

python src/eval/eval_translation.py \
    exp/bg-en/tgt.emb.txt exp/bg-en/src.trans.emb.txt \
    -d data/crosslingual/dictionaries/en-bg.5000-6500.txt \
    --cuda;
