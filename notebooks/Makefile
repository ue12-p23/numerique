NOTEBOOKS = $(shell git ls-files .teacher/[1-9]*)

all: book
.PHONY: all

include Makefile.book

include Makefile.style

NBNORMFLAGS = -l1 -s2 -t h1 -L Licence -S 'HTML\('
include Makefile.norm

include Makefile.prune

include Makefile.toc
