##
for i in {0.3,1.0,5.0,10.0};do
	grep '_id' exp/plda_scores_python_stft_sigma${i} >exp/plda_scores_python_all_stft_sigma${i};
	cat exp/plda_scores_python_target_stft_sigma${i} >>exp/plda_scores_python_all_stft_sigma${i};
done
