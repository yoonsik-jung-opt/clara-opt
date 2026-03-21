MAIN = main
LATEX = latexmk

.PHONY: all watch clean

all:
	$(LATEX) -xelatex -interaction=nonstopmode $(MAIN).tex

watch:
	$(LATEX) -xelatex -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	$(LATEX) -C $(MAIN).tex
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.synctex.gz *.toc *.xdv
