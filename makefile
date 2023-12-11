FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

p: p_nbody.o p_compute.o
	nvcc ${FLAGS} $^ -o $@ ${LIBS}
p_nbody.o: nbody.c planets.h config.h vector.h ${ALWAYS_REBUILD}
	nvcc ${FLAGS} -c $<
p_compute.o: compute.c config.h vector.h ${ALWAYS_REBUILD}
	nvcc $(FLAGS) -c $< 

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 
