include $(top_srcdir)/cuda.mk

lib_LTLIBRARIES = libhdpipeline.la

libhdpipeline_la_SOURCES = default_params.C error.C parse_command_line.C clean_filterbank_rfi.cu get_rms.cu matched_filter.cu remove_baseline.cu find_giants.cu label_candidate_clusters.cu merge_candidates.cu pipeline.cu measure_bandpass.cu median_filter.cu matched_filter.cu baseline_normalize.cu kd_dbscan_clustering.cu build_KDtree.cu grid_hash_clustering.cu optimized_sorted_clustering.cu

nobase_include_HEADERS = hd/median_filter.h hd/error.h hd/types.h hd/baseline_normalize.h hd/kd_dbscan_clustering.cuh hd/build_KDtree.cuh 

# To find the dedisp lib from bbarsdell
AM_CPPFLAGS = -I$(top_srcdir) \
	      -I$(top_srcdir)/Network \
	      -I$(top_srcdir)/Formats \
	      @DEDISP_CFLAGS@ @THRUST_CFLAGS@ @CUDA_CFLAGS@ $(BOOST_CPPFLAGS)

# 添加OpenMP支持
AM_CXXFLAGS = -fopenmp
AM_CFLAGS = -fopenmp
libhdpipeline_la_LDFLAGS = -lgomp

# 添加CUDA编译标志支持扩展Lambda和relaxed constexpr
CUDA_NVCC_FLAGS = --default-stream per-thread -Xcompiler "-fopenmp" -O2 -arch=compute_86 -code=sm_86 --extended-lambda --expt-relaxed-constexpr

include $(top_srcdir)/config/Makefile.targets
