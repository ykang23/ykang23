# command for cleaning up the testset
# make clean 
.PHONY: clean analysisset

clean:
	rm -f CU/*jpg
	rm -f non-CU/*jpg
	rm -f analysis_set_annotations.csv

analysisset:
	python3 analysisset.py /export/groups/kino/data/kino analysisset
