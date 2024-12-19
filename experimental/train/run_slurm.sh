
# Train general-domain model from scratch
# We first try learning rate 1e-3, then lower to 8e-4 if the loss diverges

ln -s /home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models HIR

bash run_pretrain_pipeline_general_without_slurm.sh \
      wiki_and_books \
      dsir_scratch \
      60200 \
      HIR/experimental/output/wiki_and_books/retrieved/retrieved_1700000_pack.jsonl \
     "true" "true" "--from_scratch --adam_beta1 0.9 --adam_beta2 0.98 --adam_eps 1e-6 --max_grad_norm 1.0" 8e-4
