# download monolingual word vectors
for LANG in bg en
do
    curl -Lo wiki.$LANG.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.$LANG.vec
done

## Downloading en-{} or {}-en dictionaries
lgs="bg"
mkdir -p crosslingual/dictionaries/
for lg in ${lgs}
do
  echo $lg
  for suffix in .txt .0-5000.txt .5000-6500.txt
  do
    fname=en-$lg$suffix
    echo "$aws_path"/dictionaries/$fname
    curl -Lo crosslingual/dictionaries/$fname https://s3.amazonaws.com/arrival/dictionaries/$fname
    fname=$lg-en$suffix
    curl -Lo crosslingual/dictionaries/$fname https://s3.amazonaws.com/arrival/dictionaries/$fname
  done
done
