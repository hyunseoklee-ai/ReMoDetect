set -e  # exit if error

# prepare folders
exp_path=exp
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="pubmed writing xsum"
source_models="gpt-3.5-turbo gpt-4 llama claude gemini"

# evalute ReMoDetect 
ReMoDetect="trained/trained_model"
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating ReMoDetect: ${SM} on ${D}_${M} ...
    python scripts/supervised.py --reward --model_name $ReMoDetect --dataset $D \
                        --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    done
  done
done


# evaluate Fast-DetectGPT in the black-box setting
settings="gpt-j-6B:gpt-neo-2.7B"
for M in $source_models; do
  for D in $datasets; do
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/fast_detect_gpt.py --reference_model_name $M1 --scoring_model_name $M2 --discrepancy_analytic \
                          --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done

# evaluate supervised detectors
supervised_models="roberta-base-openai-detector roberta-large-openai-detector"
for M in $source_models; do
  for D in $datasets; do
    for SM in $supervised_models; do
      echo `date`, Evaluating ${SM} on ${D}_${M} ...
      python scripts/supervised.py --model_name $SM --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    done
  done
done

# evaluate baselines
scoring_models="gpt-neo-2.7B"
for M in $source_models; do
  for D in $datasets; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating baseline methods on ${D}_${M}.${M2} ...
      python scripts/baselines.py --scoring_model_name ${M2} --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M2}
    done
  done
done

# evaluate DetectGPT and DetectLLM
scoring_models="gpt-neo-2.7B"
for M in $source_models; do
  for D in $datasets; do
    M1=t5-3b  # perturbation model
    for M2 in $scoring_models; do
      echo `date`, Evaluating DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_gpt.py --mask_filling_model_name ${M1} --scoring_model_name ${M2} --n_perturbations 100 --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
      # we leverage DetectGPT to generate the perturbations
      echo `date`, Evaluating DetectLLM methods on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_llm.py --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}_${M}.${M1}.perturbation_100 --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done


# evaluate GPTZero
for M in $source_models; do
  for D in $datasets; do
    echo `date`, Evaluating GPTZero on ${D}_${M} ...
    python scripts/gptzero.py --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# report eval results
python scripts/report_results.py --result_path ./exp/results/