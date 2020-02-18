voxceleb1_trials=$1
scoresfile=$2

eer=`compute-eer <(python3 local/prepare_for_eer.py $voxceleb1_trials $scoresfile) 2> /dev/null`
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $scoresfile $voxceleb1_trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $scoresfile $voxceleb1_trials 2> /dev/null`
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
