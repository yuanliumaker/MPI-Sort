all : 
	for d in lib test ; \
	do \
		$(MAKE) -C $$d ; \
	done

clean : 
	for d in lib test ; \
	do \
		$(MAKE) -C $$d clean; \
	done

.PHONY : all clean
