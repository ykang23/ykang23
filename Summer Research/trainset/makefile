# command for cleaning up the testset
# make clean 
.PHONY: cleant trainset

clean_trainset:
	rm -f CU/*jpg
	rm -f non-CU/*jpg
	rm -f train_set_annotations.csv

trainset:
	python3 trainset.py /export/groups/kino/data/kino trainset
