echo `date`, Setup the environment ...
set -e  # exit if error

exp_path=exp
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path
datasets="xsum writing pubmed"
source_models="gpt4turbo llama gemini claude"

openai_key=0
openai_base=0

export ANTHROPIC_API_KEY=xxxx
export GROQ_API_KEY=xxxx
export GOOGLE_API_KEY=xxxx
export AZURE_API_KEY=xxxx

for M in gemini; do
  for D in $datasets; do
    echo `date`, Preparing dataset ${D} by sampling from openai/${M} ...
    python scripts/data_builder.py --openai_model $M --openai_key $openai_key --openai_base $openai_base \
                --dataset $D --n_samples 150 --do_temperature --temperature 0.8 --batch_size 1 \
                --output_file $data_path/${D}_${M}
  done
done
