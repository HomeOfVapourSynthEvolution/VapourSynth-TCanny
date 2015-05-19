#----------------------------------------------------------------------------
#  Makefile for VapourSynth-TCanny
#----------------------------------------------------------------------------

include config.mak

vpath %.cpp $(SRCDIR)
vpath %.h $(SRCDIR)

SRCS = TCanny/TCanny.cpp

OBJS = $(SRCS:%.cpp=%.o)

.PHONY: all install clean distclean dep

all: $(LIBNAME)

$(LIBNAME): $(OBJS)
	$(LD) -o $@ $(LDFLAGS) $^ $(LIBS)
	-@ $(if $(STRIP), $(STRIP) -x $@)

%.o: %.cpp .depend
	$(CXX) $(CXXFLAGS) -c $< -o $@

install: all
	install -d $(libdir)
	install -m 755 $(LIBNAME) $(libdir)

clean:
	$(RM) *.dll *.so *.dylib $(OBJS) .depend

distclean: clean
	$(RM) config.*

dep: .depend

ifneq ($(wildcard .depend),)
include .depend
endif

.depend: config.mak
	@$(RM) .depend
	@$(foreach SRC, $(SRCS:%=$(SRCDIR)/%), $(CXX) $(SRC) $(CXXFLAGS) -MT $(SRC:$(SRCDIR)/%.cpp=%.o) -MM >> .depend;)

config.mak:
	./configure
