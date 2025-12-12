include $(top_srcdir)/cuda.mk

include_HEADERS = 

bin_PROGRAMS = heimdall coincidencer coincidencer_client candidate_profiler fil2pgm generate_dmlist heimdall_pl heimdall_async heimdall_sync

AM_CXXFLAGS = \
  -I$(top_srcdir) \
	-I$(top_srcdir)/Formats \
	-I$(top_srcdir)/Network \
	-I$(top_srcdir)/Pipeline \
	@PSRDADA_CFLAGS@ \
	@DEDISP_CFLAGS@ \
	-fopenmp

heimdall_SOURCES = heimdall.C
coincidencer_SOURCES = coincidencer.C Candidates.C
coincidencer_client_SOURCES = coincidencer_client.C
candidate_profiler_SOURCES = candidate_profiler.C
fil2pgm_SOURCES = fil2pgm.C
generate_dmlist_SOURCES = generate_dmlist.C
heimdall_pl_SOURCES = heimdall_pl.C
heimdall_async_SOURCES = heimdall_async.C
heimdall_sync_SOURCES = heimdall_sync.C

LDADD = \
  $(top_builddir)/Formats/libhdformats.la \
  $(top_builddir)/Pipeline/libhdpipeline.la \
  $(top_builddir)/Network/libhdnetwork.la \
  @CUDA_LIBS@ @DEDISP_LIBS@ @PSRDADA_LIBS@ \
  -lgomp

heimdall_CFLAGS = $(CUDA_CFLAGS)
heimdall_async_CFLAGS = $(CUDA_CFLAGS)
heimdall_sync_CFLAGS = $(CUDA_CFLAGS)
generate_dmlist_CXXFLAGS = $(AM_CXXFLAGS) @DEDISP_CFLAGS@

include $(top_srcdir)/config/Makefile.targets

