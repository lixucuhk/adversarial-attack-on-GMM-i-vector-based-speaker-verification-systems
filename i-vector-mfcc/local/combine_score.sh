##
for i in {0.3,1.0,5.0,10.0,20.0,30.0,50.0};do
	grep '_id' exp/score_adv_stft_sigma${i} >exp/score_all_adv_stft_sigma${i};
	cat exp/score_target_adv_stft_sigma${i} >>exp/score_all_adv_stft_sigma${i};
done
