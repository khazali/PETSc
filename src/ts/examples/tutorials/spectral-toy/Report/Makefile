FILE = SEM_Petsc
BIBS = SEM_Petsc.bib

$(FILE).pdf: $(FILE).tex $(BIBS)
	pdflatex $(FILE)
	bibtex $(FILE)
	pdflatex $(FILE)
	bibtex $(FILE)
	pdflatex $(FILE)

clean:
	rm -f  $(FILE).pdf $(FILE).aux $(FILE).bbl $(FILE).blg $(FILE).log $(FILE).out $(FILE).toc $(FILE).ind $(FILE).ilg $(FILE).cb $(FILE).idx $(FILE).dvi $(FILE).ps *~

