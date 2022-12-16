# Bilingual Offline Transcriptions

## Usage

This file provides instructions for running each of the experiments in the
project. All of the experiments were done by modifying core ESPnet code and do
not include separate scripts.

## Directory structure

Before running any of the experiments,
[install ESPnet from source](https://espnet.github.io/espnet/installation.html)
and copy the contents of _asr1_ to _egs2/commonvoice/asr1_ with directory
structure intact. This can be done with

```
rsync asr1 $ESPNET_DIR/egs2/commonvoice/asr1
```

## Hyperparameters and configuration

All configuration, including modified hyperparameters, are written into the
files under _asr1/conf/tuning_. In particular,
_train_asr_conformer5_linear-units-256-output-64-num-blocks-6.yaml_ contains
the baseline configuration used by all experiments.

# Experiment instructions

## Baseline

### Download data

We use the Common Voice 11.0 dataset rather than the Common Voice 5.1 dataset
that ESPnet2 uses by default, so a few tweaks to the script are necessary.

1.  Download the Lithuanian and Slovak versions of Common Voice Corpus 11.0
    from https://commonvoice.mozilla.org/en/datasets to the following locations:
    - _egs2/commonvoice/asr1/downloads/lt.tar.gz_
    - _egs2/commonvoice/asr1/downloads/sk.tar.gz_
2.  Change line 48 of _egs2/commonvoice/asr1/local/download_and_untar.sh_ from
    `size_ok=false` to `size_ok=true` to force the script to accept the
    archives you just downloaded.
3.  Change the directory name in line 57 of
    _egs2/commonvoice/asr1/local/data.sh_ from
    `cv-corpus-5.1-2020-06-22` to `cv-corpus-11.0-2022-09-21`.

### Augment and combine data

```
for dataset_lang in lt sk
  ./asr.sh --stage 1 --stop_stage 2 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --local_data_opts "--lang $dataset_lang" --speed_perturb_factors "0.9 1.0 1.1"
done
utils/combine_data.sh data/train_lt\+sk_sp data/train_lt_sp data/train_sk_sp
utils/combine_data.sh data/dev_lt\+sk data/dev_lt data/dev_sk
utils/combine_data.sh data/test_lt\+sk data/test_lt data/test_sk
```

### Prepare data

```
dataset_lang=lt\+sk
./asr.sh --stage 3 --stop_stage 4 --nj 4 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --speed_perturb_factors "0.9 1.0 1.1" --audio_format "flac" --lm_train_text "data/train_${dataset_lang}_sp/text"
./asr.sh --stage 5 --stop_stage 5 --nj 1 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --token_type bpe --nbpe 150 --bpemode "unigram" --bpe_train_text "data/train_${dataset_lang}_sp/text"
```

### Train and evaluate

```
config=train_asr_conformer5_linear-units-256-output-64-num-blocks-6
./asr.sh --stage 10 --stop_stage 10 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --speed_perturb_factors "0.9 1.0 1.1" --nj 1 --token_type bpe --nbpe 150 --asr_config "conf/tuning/$config.yaml"
./asr.sh --stage 11 --stop_stage 11 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --speed_perturb_factors "0.9 1.0 1.1" --nj 1 --token_type bpe --nbpe 150 --ngpu 1 --asr_config "conf/tuning/$config.yaml"
./asr.sh --inference_nj 1 --stage 12 --stop_stage 13 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --speed_perturb_factors "0.9 1.0 1.1" --token_type bpe --nbpe 150 --gpu_inference true --asr_config "conf/tuning/$config.yaml" --use_lm false
```

## Language marker in text (explicit)

### Set up directories

```
cp -RT exp/asr_stats_raw_${dataset_lang}_bpe150_sp exp/asr_stats_raw_${dataset_lang}_bpe150_sp_bak
cp -RT exp/asr_${config}_raw_${dataset_lang}_bpe150_sp exp/asr_${config}_raw_${dataset_lang}_bpe150_sp_bak
```

### Inject new tokens in preprocessing

1.  Change the signature of `CommonPreprocessor._text_process` in _espnet2/train/preprocessor.py_ to add the parameter `uid: Union[str, None] = None`
2.  Add the following lines between the definitions of `tokens` and `text_ints` in the definition of `CommonPreprocessor._text_process` in _espnet2/train/preprocessor.py_:
    ```
    if "_lt_" in uid:
        tokens.insert(0, "<lt>")
    elif "_sk_" in uid:
        tokens.insert(0, "<sk>")
    else:
        raise NotImplementedError(uid)
    ```
3.  Change the callsite of `_text_process` in `CommonPreprocessor.__call__` in _espnet2/train/preprocessor.py_ from `data = self._text_process(data)` to `data = self._text_process(data, uid)`
4.  Run the rest of the _Baseline_ instructions from stage 5 with `--bpe_nlsyms \<lt\>,\<sk\>`.

### Cleanup

1.  Comment out the changes in step 2 of _Inject new tokens in preprocessing_.
2.  Rerun stage 5 of the _Baseline_ instructions.

```
mv -T exp/asr_stats_raw_${dataset_lang}_bpe150_sp exp/asr_stats_raw_${dataset_lang}_text-marker-explicit_bpe150_sp
mv -T exp/asr_${config}_raw_${dataset_lang}_bpe150_sp exp/asr_${config}_raw_${dataset_lang}_text-marker-explicit_bpe150_sp
mv -T exp/asr_stats_raw_${dataset_lang}_bpe150_sp_bak exp/asr_stats_raw_${dataset_lang}_bpe150_sp
mv -T exp/asr_${config}_raw_${dataset_lang}_bpe150_sp_bak exp/asr_${config}_raw_${dataset_lang}_bpe150_sp
```

## Language marker in text (BPE)

### Set up directories

```
dataset_lang=lt\+sk_text-marker-bpe
cp -RT "dump/raw/train_lt+sk_sp" "dump/raw/train_${dataset_lang}_sp"
cp -RT "dump/raw/dev_lt+sk" "dump/raw/dev_$dataset_lang"
cp -RT "dump/raw/test_lt+sk" "dump/raw/test_$dataset_lang"
```

### Inject new tokens in data

```
sed -Ei 's/(_([a-z][a-z])_[^ ]+ )/\1<\2>/' "dump/raw/train_${dataset_lang}_sp/text"
sed -Ei 's/(_([a-z][a-z])_[^ ]+ )/\1<\2>/' "dump/raw/dev_${dataset_lang}/text"
sed -Ei 's/(_([a-z][a-z])_[^ ]+ )/\1<\2>/' "dump/raw/test_${dataset_lang}/text"
```

Run the rest of the _Baseline_ instructions from stage 5 with `--bpe_train_text data/train_lt\+sk_sp/text --bpe_nlsyms \<lt\>,\<sk\>`.

### Cleanup

```
dataset_lang=lt\+sk
```

## Language marker in speech (single byte)

### Inject new markers in preprocessing

1.  Change the signature of `CommonPreprocessor._speech_process` in _espnet2/train/preprocessor.py_ to add the parameter `uid: Union[str, None] = None`
2.  Add the following lines to the end of the `if self.speech_name in data` block in _espnet2/train/preprocessor.py_:
    ```
    if "_lt_" in uid:
        data[self.speech_name] = np.r_[1, data[self.speech_name]]
    elif "_sk_" in uid:
        data[self.speech_name] = np.r_[-1, data[self.speech_name]]
    else:
        raise NotImplementedError(uid)
    ```
3.  Change the callsite of `_speech_process` in `CommonPreprocessor.__call__` in _espnet2/train/preprocessor.py_ from `data = self._speech_process(data)` to `data = self._speech_process(data, uid)`
4.  Run the rest of the _Baseline_ instructions from stage 10.

### Cleanup

Comment out the changes in step 2 of _Inject new tokens in preprocessing_.

## Language marker in speech (audio clip)

### Set up directories

```
dataset_lang=lt\+sk_speech-marker-clip
cp -RT "data/train_lt+sk_sp" "data/train_${dataset_lang}_sp"
cp -RT "data/dev_lt+sk" "data/dev_$dataset_lang"
cp -RT "data/test_lt+sk" "data/test_$dataset_lang"
```

### Inject new clips in data

```
sed -Ei 's/(_([a-z][a-z])_.*?)-i ([^ ]+)/\1-i "concat:\2.mp3|\3"/' "data/train_${dataset_lang}_sp/wav.scp"
sed -Ei 's/(_([a-z][a-z])_.*?)-i ([^ ]+)/\1-i "concat:\2.mp3|\3"/' "data/dev_${dataset_lang}/wav.scp"
sed -Ei 's/(_([a-z][a-z])_.*?)-i ([^ ]+)/\1-i "concat:\2.mp3|\3"/' "data/test_${dataset_lang}/wav.scp"
```

Run the rest of the _Baseline_ instructions from stage 10.

### Cleanup

```
dataset_lang=lt\+sk
```

## Concatenated data

### Set up directories

```
dataset_lang=lt\+sk_concatenated
for segment in train dev test; do
  cp -RT "data/${segment}_sk" "data/${segment}_$dataset_lang"
done
```

### Concatenate speech and text

```
for segment in train dev test; do
  paste "data/${segment}_sk/wav.scp" "data/${segment}_lt/wav.scp" | sed -En 's/-i ([^ ]+).*\t.*-i ([^ ]+)/-i "concat:\1|\2"/p' - > "data/${segment}_$dataset_lang/wav.scp"
  paste "data/${segment}_sk/text" "data/${segment}_lt/text" | sed -En 's/(.)\t([^ ]+)/\1/p' > "data/${segment}_${dataset_lang}/text"
  sed -i $(wc -l < "data/${segment}_$dataset_lang/text")q "data/${segment}_$dataset_lang/utt2spk"
done
```

### Augment concatenated data

```
./asr.sh --stage 2 --stop_stage 2 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --local_data_opts "--lang $dataset_lang" --speed_perturb_factors "0.9 1.0 1.1"
```

Run the rest of the _Baseline_ instructions from stage 3.

### Cleanup

```
dataset_lang=lt\+sk
```

## Concatenated data with language marker in text (BPE)

### Set up directories

```
dataset_lang=lt\+sk_concatenated_text-marker-bpe
for segment in train dev test; do
  cp -RT "data/${segment}_sk" "data/${segment}_$dataset_lang"
done
```

### Concatenate speech and text with new tokens

```
for segment in train dev test; do
  paste "data/${segment}_sk/wav.scp" "data/${segment}_lt/wav.scp" | sed -En 's/-i ([^ ]+).*\t.*-i ([^ ]+)/-i "concat:\1|\2"/p' - > "data/${segment}_$dataset_lang/wav.scp"
  paste "data/${segment}_sk/text" "data/${segment}_lt/text" | sed -En 's/^([^ ]+) (.+)\t([^ ]+ )/\1 <sk>\2 <lt>/p' > "data/${segment}_${dataset_lang}/text"
  sed -i $(wc -l < "data/${segment}_$dataset_lang/text")q "data/${segment}_$dataset_lang/utt2spk"
done
```

### Augment concatenated data with new tokens

```
./asr.sh --stage 2 --stop_stage 2 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --local_data_opts "--lang $dataset_lang" --speed_perturb_factors "0.9 1.0 1.1"
```

Run the rest of the _Baseline_ instructions from stage 3 with `--bpe_train_text data/train_lt\+sk_concatenated_sp/text --bpe_nlsyms \<lt\>,\<sk\>`.

### Cleanup

```
dataset_lang=lt\+sk
```

## Concatenated data with language marker in speech (audio clip)

### Set up directories

```
dataset_lang=lt\+sk_concatenated_speech-marker-clip
for segment in train dev test; do
  cp -RT "data/${segment}_sk" "data/${segment}_$dataset_lang"
done
```

### Concatenate speech and text with new clips

```
for segment in train dev test; do
  paste "data/${segment}_sk/wav.scp" "data/${segment}_lt/wav.scp" | sed -En 's/-i ([^ ]+).*\t.*-i ([^ ]+)/-i "concat:sk.mp3|\1|lt.mp3|\2"/p' - > "data/${segment}_$dataset_lang/wav.scp"
  paste "data/${segment}_sk/text" "data/${segment}_lt/text" | sed -En 's/(.)\t([^ ]+)/\1/p' > "data/${segment}_${dataset_lang}/text"
  sed -i $(wc -l < "data/${segment}_$dataset_lang/text")q "data/${segment}_$dataset_lang/utt2spk"
done
```

### Augment concatenated data with new clips

```
./asr.sh --stage 2 --stop_stage 2 --train_set "train_$dataset_lang" --valid_set "dev_$dataset_lang" --test_sets "dev_$dataset_lang test_$dataset_lang" --lang "$dataset_lang" --local_data_opts "--lang $dataset_lang" --speed_perturb_factors "0.9 1.0 1.1"
```

Run the rest of the _Baseline_ instructions from stage 3.

### Cleanup

```
dataset_lang=lt\+sk
```
