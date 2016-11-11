.SUFFIXES: .c .o .swg 

include ../../makefile.inc

CC = gcc
CC1 = g++
CFLAGS1 = -g -Wall -Wno-sign-compare

all: pqtrain pqsearch clean


ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "yes"
  EXTRAMATRIXFLAG+=-fopenmp
  EXTRAYAELLDFLAG+=-fopenmp
endif

#############################
# Various  


.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@ $(FLAGS) $(EXTRACFLAGS) $(YAELCFLAGS) 

pqtrain: train_main.o dataset.o database.o model.o 
	$(CC1) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS) 

pqsearch: main.o pqsearchengine.o database.o model.o
	$(CC1) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS)  


#############################
# Dependencies  

# for i in *.c ; do cpp -I../.. -MM $i; done
main.o: pqsearchengine.h database.h model.h 
train_main.o: dataset.h database.h model.h 
dataset.o: dataset.h featureflfmt.h ../../yael/vector.h ../../yael/nn.h ../../yael/kmeans.h ../../yael/sorting.h ../../yael/machinedeps.h
database.o: database.h util.h types.h ../../yael/vector.h ../../yael/nn.h ../../yael/kmeans.h ../../yael/sorting.h ../../yael/machinedeps.h
model.o: model.h featureflfmt.h dataset.h database.h types.h ../../yael/vector.h ../../yael/nn.h ../../yael/kmeans.h ../../yael/sorting.h ../../yael/machinedeps.h  
pqsearchengine.o: pqsearchengine.h featureflfmt.h dataset.h database.h model.h types.h util.h ../../yael/vector.h ../../yael/nn.h ../../yael/kmeans.h ../../yael/sorting.h ../../yael/machinedeps.h 

.PHONY: clean
clean:
	rm -f *.o


