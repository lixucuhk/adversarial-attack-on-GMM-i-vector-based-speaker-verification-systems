##
for i in {0.3,1.0,5.0,10.0,20.0,30.0,50.0};do
	grep '_id' exp/scores_adv_mfcc_sigma${i} >exp/scores_all_adv_mfcc_sigma${i};
	cat exp/scores_target_adv_mfcc_sigma${i} >>exp/scores_all_adv_mfcc_sigma${i};
done
