# command for cleaning up the testset
# make clean 
.PHONY: clean testset

clean:
	rm -f CU/*jpg
	rm -f non-CU/*jpg
	rm -f test_set_annotations.csv
testset:
	python3 testset.py /export/groups/kino/data/kino testset
