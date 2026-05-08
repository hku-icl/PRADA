nohup scripts/serve_draft_model.sh > draft.log 2>&1 &
nohup scripts/serve_target_model.sh > target.log 2>&1 &
nohup scripts/serve_prm.sh > prm.log 2>&1 &