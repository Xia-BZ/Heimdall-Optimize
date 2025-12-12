/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
using std::cerr;
using std::cout;
using std::endl;
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
using thrust::device_vector;
using thrust::host_vector;
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/version.h>
#include <typeinfo>

#include "hd/clean_filterbank_rfi.h"
#include "hd/maths.h"
#include "hd/pipeline.h"

#include "hd/baseline_normalize.h"
#include "hd/find_giants.h"
#include "hd/get_rms.h"
#include "hd/label_candidate_clusters.h"
#include "hd/matched_filter.h"
#include "hd/merge_candidates.h"
#include "hd/remove_baseline.h"

#include "hd/ClientSocket.h"
#include "hd/DataSource.h"
#include "hd/SocketException.h"
#include "hd/stopwatch.h" // For benchmarking
// #include "hd/write_time_series.h" // For debugging
#include "hd/cached_allocator.cuh"
#include "hd/kd_dbscan_clustering.cuh"

#include <dedisp.h>
#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("CUDA Error:\n");                                                 \
      printf("    File:       %s\n", __FILE__);                                \
      printf("    Line:       %d\n", __LINE__);                                \
      printf("    Error code: %d\n", error_code);                              \
      printf("    Error text: %s\n", cudaGetErrorString(error_code));          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define HD_BENCHMARK

#ifdef HD_BENCHMARK
void start_timer(Stopwatch &timer) { timer.start(); }
void stop_timer(Stopwatch &timer) {
  cudaDeviceSynchronize();
  timer.stop();
}
#else
void start_timer(Stopwatch &timer) {}
void stop_timer(Stopwatch &timer) {}
#endif // HD_BENCHMARK

#include <utility> // For std::pair
template <typename T, typename U> std::pair<T &, U &> tie(T &a, U &b) {
  return std::pair<T &, U &>(a, b);
}

// static const int gpu_idx = 4; // set GPU id
#define THRUST_DEBUG 1

struct hd_pipeline_t {
  hd_params params;
  dedisp_plan dedispersion_plan;
  // MPI_Comm    communicator;
  // const int N_threads = 1;

  // Memory buffers used during pipeline execution
  // std::vector<hd_byte> h_clean_filterbank;
  device_vector<hd_byte> h_clean_filterbank;
  host_vector<hd_byte> h_dm_series;

  std::vector<device_vector<hd_float>> d_time_series;
  std::vector<device_vector<hd_float>> d_filtered_series;

  // Persistent DM series buffer to avoid repeated allocation
  hd_byte *d_dm_series;
  size_t d_dm_series_allocated_size;

  // device_vector<hd_float> d_time_series;
  // device_vector<hd_float> d_filtered_series;
  // hd_float *d_time_series;
  // hd_float *d_filtered_series;
};

hd_error allocate_gpu(const hd_pipeline pl) {
  // TODO: This is just a simple proc-->GPU heuristic to get us started
  // int gpu_count;
  // cudaGetDeviceCount(&gpu_count);
  // int proc_idx;
  // MPI_Comm comm = pl->communicator;
  // MPI_Comm_rank(comm, &proc_idx);
  int proc_idx = pl->params.beam;
  int gpu_idx = pl->params.gpu_id;
  // for test

  cudaError_t cerror = cudaSetDevice(gpu_idx);
  if (cerror != cudaSuccess) {
    cerr << "Could not setCudaDevice to " << gpu_idx << ": "
         << cudaGetErrorString(cerror) << endl;
    return throw_cuda_error(cerror);
  }

  if (pl->params.verbosity >= 1) {
    cout << "Process " << proc_idx << " using GPU " << gpu_idx << endl;
  }

  if (!pl->params.yield_cpu) {
    if (pl->params.verbosity >= 2) {
      cout << "\tProcess " << proc_idx << " setting CPU to spin" << endl;
    }
    cerror = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    if (cerror != cudaSuccess) {
      return throw_cuda_error(cerror);
    }
  } else {
    if (pl->params.verbosity >= 2) {
      cout << "\tProcess " << proc_idx << " setting CPU to yield" << endl;
    }
    // Note: This Yield flag doesn't seem to work properly.
    //   The BlockingSync flag does the job, although it may interfere
    //     with GPU/CPU overlapping (not currently used).
    // cerror = cudaSetDeviceFlags(cudaDeviceScheduleYield);
    cerror = cudaSetDeviceFlags(cudaDeviceBlockingSync);
    if (cerror != cudaSuccess) {
      return throw_cuda_error(cerror);
    }
  }

  return HD_NO_ERROR;
}

unsigned int get_filter_index(unsigned int filter_width) {
  // This function finds log2 of the 32-bit power-of-two number v
  unsigned int v = filter_width;
  static const unsigned int b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0,
                                   0xFF00FF00, 0xFFFF0000};
  register unsigned int r = (v & b[0]) != 0;
  for (int i = 4; i > 0; --i) {
    r |= ((v & b[i]) != 0) << i;
  }
  return r;
}

hd_error hd_create_pipeline(hd_pipeline *pipeline_, hd_params params) {
  *pipeline_ = 0;

  // Note: We use a smart pointer here to automatically clean up after errors
#if __cplusplus <= 199711L
  typedef std::auto_ptr<hd_pipeline_t> smart_pipeline_ptr;
#else
  typedef std::unique_ptr<hd_pipeline_t> smart_pipeline_ptr;
#endif
  smart_pipeline_ptr pipeline = smart_pipeline_ptr(new hd_pipeline_t());
  if (!pipeline.get()) {
    return throw_error(HD_MEM_ALLOC_FAILED);
  }

  pipeline->params = params;
  // 初始化DM系列缓冲区
  pipeline->d_dm_series = nullptr;
  pipeline->d_dm_series_allocated_size = 0;

  if (params.verbosity >= 2) {
    cout << "\tAllocating GPU..." << endl;
  }

  hd_error error = allocate_gpu(pipeline.get());
  if (error != HD_NO_ERROR) {
    return throw_error(error);
  }

  if (params.verbosity >= 3) {
    cout << "nchans = " << params.nchans << endl;
    cout << "dt     = " << params.dt << endl;
    cout << "f0     = " << params.f0 << endl;
    cout << "df     = " << params.df << endl;
  }

  if (params.verbosity >= 2) {
    cout << "\tCreating dedispersion plan..." << endl;
  }

  dedisp_error derror;
  derror = dedisp_create_plan(&pipeline->dedispersion_plan, params.nchans,
                              params.dt, params.f0, params.df);
  if (derror != DEDISP_NO_ERROR) {
    return throw_dedisp_error(derror);
  }
  // TODO: Consider loading a pre-generated DM list instead for flexibility
  derror = dedisp_generate_dm_list(
      pipeline->dedispersion_plan, pipeline->params.dm_min,
      pipeline->params.dm_max, pipeline->params.dm_pulse_width,
      pipeline->params.dm_tol);
  if (derror != DEDISP_NO_ERROR) {
    return throw_dedisp_error(derror);
  }

  if (pipeline->params.use_scrunching) {
    derror = dedisp_enable_adaptive_dt(pipeline->dedispersion_plan,
                                       pipeline->params.dm_pulse_width,
                                       pipeline->params.scrunch_tol);
    if (derror != DEDISP_NO_ERROR) {
      return throw_dedisp_error(derror);
    }
  }

  *pipeline_ = pipeline.release();

  if (params.verbosity >= 2) {
    cout << "\tInitialisation complete." << endl;
  }

  if (params.verbosity >= 1) {
    cout << "Using Thrust v" << THRUST_MAJOR_VERSION << "."
         << THRUST_MINOR_VERSION << "." << THRUST_SUBMINOR_VERSION << endl;
  }

  return HD_NO_ERROR;
}

hd_error hd_execute(hd_pipeline pl, const hd_byte *h_filterbank, hd_size nsamps,
                    hd_size nbits, hd_size first_idx,
                    hd_size *nsamps_processed) {
  hd_error error = HD_NO_ERROR;

  Stopwatch total_timer;
  Stopwatch memory_timer;
  Stopwatch clean_timer;
  Stopwatch dedisp_timer;
  Stopwatch communicate_timer;
  Stopwatch copy_timer;
  Stopwatch baseline_timer;
  Stopwatch normalise_timer;
  Stopwatch filter_timer;
  Stopwatch coinc_timer;
  Stopwatch giants_timer;
  Stopwatch candidates_timer;
  Stopwatch DSPT_timer;
  const unsigned int N_threads = pl->params.num_threads; // set thread number
  start_timer(total_timer);

  start_timer(clean_timer);
  // Note: Filterbank cleaning must be done out-of-place
  hd_size nbytes = nsamps * pl->params.nchans * nbits / 8;
  start_timer(memory_timer);
  pl->h_clean_filterbank.resize(nbytes);
  std::vector<int> h_killmask(pl->params.nchans, 1);
  stop_timer(memory_timer);

  if (pl->params.verbosity >= 2) {
    cout << "\tCleaning 0-DM filterbank..." << endl;
  }

  // Start by cleaning up the filterbank based on the zero-DM time series
  hd_float cleaning_dm = 0.f;
  if (pl->params.verbosity >= 3) {
    /*
    cout << "\tWriting dirty filterbank to disk..." << endl;
    write_host_filterbank(&h_filterbank[0],
                          pl->params.nchans, nsamps, nbits,
                          pl->params.dt, pl->params.f0, pl->params.df,
                          "dirty_filterbank.fil");
    */
  }
  // Note: We only clean the narrowest zero-DM signals; otherwise we
  //         start removing real stuff from higher DMs.
  error = clean_filterbank_rfi(
      pl->dedispersion_plan, &h_filterbank[0], nsamps, nbits,
      thrust::raw_pointer_cast(pl->h_clean_filterbank.data()), &h_killmask[0],
      cleaning_dm, pl->params.dt, pl->params.baseline_length,
      pl->params.rfi_tol, pl->params.rfi_min_beams, pl->params.rfi_broad,
      pl->params.rfi_narrow,
      1); // pl->params.boxcar_max);
  if (error != HD_NO_ERROR) {
    return throw_error(error);
  }

  if (pl->params.verbosity >= 2) {
    cout << "Applying manual killmasks" << endl;
  }

  error = apply_manual_killmasks(pl->dedispersion_plan, &h_killmask[0],
                                 pl->params.num_channel_zaps,
                                 pl->params.channel_zaps);
  if (error != HD_NO_ERROR) {
    return throw_error(error);
  }

  hd_size good_chan_count =
      thrust::reduce(h_killmask.begin(), h_killmask.end());
  hd_size bad_chan_count = pl->params.nchans - good_chan_count;
  if (pl->params.verbosity >= 2) {
    cout << "Bad channel count = " << bad_chan_count << endl;
  }

  // TESTING
  // h_clean_filterbank.assign(h_filterbank, h_filterbank+nbytes);

  stop_timer(clean_timer);

  if (pl->params.verbosity >= 3) {
    /*
    cout << "\tWriting killmask to disk..." << endl;
    std::ofstream killfile("killmask.dat");
    for( size_t i=0; i<h_killmask.size(); ++i ) {
      killfile << h_killmask[i] << "\n";
    }
    killfile.close();

    cout << "\tWriting cleaned filterbank to disk..." << endl;
    write_host_filterbank(&pl->h_clean_filterbank[0],
                          pl->params.nchans, nsamps, nbits,
                          pl->params.dt, pl->params.f0, pl->params.df,
                          "clean_filterbank.fil");
    */
  }
  if (pl->params.verbosity >= 2) {
    cout << "\tGenerating DM list..." << endl;
  }

  if (pl->params.verbosity >= 3) {
    cout << "dm_min = " << pl->params.dm_min << endl;
    cout << "dm_max = " << pl->params.dm_max << endl;
    cout << "dm_tol = " << pl->params.dm_tol << endl;
    cout << "dm_pulse_width = " << pl->params.dm_pulse_width << endl;
    cout << "nchans = " << pl->params.nchans << endl;
    cout << "dt = " << pl->params.dt << endl;

    cout << "dedisp nchans = "
         << dedisp_get_channel_count(pl->dedispersion_plan) << endl;
    cout << "dedisp dt = " << dedisp_get_dt(pl->dedispersion_plan) << endl;
    cout << "dedisp f0 = " << dedisp_get_f0(pl->dedispersion_plan) << endl;
    cout << "dedisp df = " << dedisp_get_df(pl->dedispersion_plan) << endl;
  }

  hd_size dm_count = dedisp_get_dm_count(pl->dedispersion_plan);
  // cout << "DM count = " << dm_count << endl;
  const float *dm_list = dedisp_get_dm_list(pl->dedispersion_plan);

  const dedisp_size *scrunch_factors =
      dedisp_get_dt_factors(pl->dedispersion_plan);
  if (pl->params.verbosity >= 3) {
    cout << "DM List for " << pl->params.dm_min << " to " << pl->params.dm_max
         << endl;
    for (hd_size i = 0; i < dm_count; ++i) {
      cout << dm_list[i] << endl;
    }
  }

  if (pl->params.verbosity >= 2) {
    cout << "Scrunch factors:" << endl;
    for (hd_size i = 0; i < dm_count; ++i) {
      cout << scrunch_factors[i] << " ";
    }
    cout << endl;
  }

  // Set channel killmask for dedispersion
  dedisp_set_killmask(pl->dedispersion_plan, &h_killmask[0]);
  if (dedisp_get_max_delay(pl->dedispersion_plan) > nsamps) {
    cerr << "maximum DM delay=" << dedisp_get_max_delay(pl->dedispersion_plan)
         << endl;
    cerr << "Number of samples=" << nsamps << endl;
    return throw_error(HD_TOO_FEW_NSAMPS);
  }

  hd_size nsamps_computed =
      nsamps - dedisp_get_max_delay(pl->dedispersion_plan);
  hd_size series_stride = nsamps_computed;

  // Report the number of samples that will be properly processed
  *nsamps_processed = nsamps_computed - pl->params.boxcar_max;

  if (pl->params.verbosity >= 3) {
    cout << "dm_count = " << dm_count << endl;
    cout << "max delay = " << dedisp_get_max_delay(pl->dedispersion_plan)
         << endl;
    cout << "nsamps_computed = " << nsamps_computed << endl;
  }

  hd_size beam = pl->params.beam;

  if (pl->params.verbosity >= 2) {
    cout << "\tAllocating memory for pipeline computations..." << endl;
  }

  start_timer(memory_timer);
  // 分配缓冲区内存
  pl->d_time_series.resize(N_threads);
  pl->d_filtered_series.resize(N_threads);
  // pl->h_dm_series.resize(series_stride * pl->params.dm_nbits / 8 * dm_count);
  for (int i = 0; i < N_threads; ++i) {
    pl->d_time_series[i].resize(series_stride);
    pl->d_filtered_series[i].resize(series_stride, 0);
  }

  std::vector<RemoveBaselinePlan> baseline_removers(N_threads);
  std::vector<GetRMSPlan> rms_getters(N_threads);
  std::vector<MatchedFilterPlan<hd_float>> matched_filter_plans(N_threads);
  std::vector<GiantFinder> giant_finders(N_threads);

  RemoveBaselinePlan baseline_remover;
  GetRMSPlan rms_getter;

  MatchedFilterPlan<hd_float> matched_filter_plan;
  GiantFinder giant_finder;
  std::vector<thrust::device_vector<hd_float>> d_giant_peaks_t(N_threads);
  std::vector<thrust::device_vector<hd_size>> d_giant_inds_t(N_threads);
  std::vector<thrust::device_vector<hd_size>> d_giant_begins_t(N_threads);
  std::vector<thrust::device_vector<hd_size>> d_giant_ends_t(N_threads);

  std::vector<thrust::device_vector<hd_size>> d_giant_filter_inds_t(N_threads);
  std::vector<thrust::device_vector<hd_size>> d_giant_dm_inds_t(N_threads);
  std::vector<thrust::device_vector<hd_size>> d_giant_members_t(N_threads);

  typedef thrust::device_ptr<hd_float> dev_float_ptr;
  typedef thrust::device_ptr<hd_size> dev_size_ptr;
  stop_timer(memory_timer);

  if (pl->params.verbosity >= 2) {
    cout << "\tDedispersing for DMs " << dm_list[0] << " to "
         << dm_list[dm_count - 1] << "..." << endl;
  }
  // 初始化基线移除和归一化计划
  std::vector<RemoveBaselineAndNormalizePlan> baseline_normalizer(N_threads);
  // for (int i = 0; i < N_threads; ++i)
  // {
  //   baseline_normalizer[i].init(nsamps_computed);
  // }

  // Dedisperse
  // 使用pipeline中的持久化缓冲区，避免重复分配
  const size_t dm_series_size =
      series_stride * pl->params.dm_nbits / 8 * dm_count;
  const size_t required_bytes = dm_series_size * sizeof(hd_byte);

  // 检查是否需要重新分配（仅在需要更大空间时）
  if (pl->d_dm_series == nullptr ||
      pl->d_dm_series_allocated_size < required_bytes) {
    // 释放旧的内存（如果存在）
    if (pl->d_dm_series != nullptr) {
      cudaFree(pl->d_dm_series);
    }

    // 分配更大的内存，预留一些空间以避免频繁重新分配
    size_t allocation_size = required_bytes * 1.2; // 20%的缓冲
    CHECK(cudaMallocManaged(&pl->d_dm_series, allocation_size));
    pl->d_dm_series_allocated_size = allocation_size;

    if (pl->params.verbosity >= 2) {
      cout << "重新分配DM系列缓冲区: "
           << static_cast<double>(allocation_size) / (1024 * 1024 * 1024)
           << " GB" << endl;
    }
  }

  hd_byte *d_dm_series = pl->d_dm_series;

  // cudaMemAdvise(d_dm_series, dm_series_size * sizeof(hd_byte),
  //               cudaMemAdviseSetPreferredLocation, 0);
  // cudaMemAdvise(d_dm_series, dm_series_size * sizeof(hd_byte),
  //               cudaMemAdviseSetReadMostly, 0);
  dedisp_error derror;
  // const dedisp_byte *in = &pl->h_clean_filterbank[0];
  const dedisp_byte *in =
      thrust::raw_pointer_cast(pl->h_clean_filterbank.data());
  // dedisp_byte *out = &pl->h_dm_series[0];
  dedisp_byte *out = d_dm_series;
  dedisp_size in_nbits = nbits;
  dedisp_size in_stride = pl->params.nchans * in_nbits / 8;
  dedisp_size out_nbits = pl->params.dm_nbits;
  dedisp_size out_stride = series_stride * out_nbits / 8;
  unsigned flags = DEDISP_DEVICE_POINTERS;

  start_timer(dedisp_timer);
  derror = dedisp_execute_adv(pl->dedispersion_plan, nsamps, in, in_nbits,
                              in_stride, out, out_nbits, out_stride, flags);
  if (derror != DEDISP_NO_ERROR) {
    printf("\nERROR: Failed to execute dedispersion plan: %s\n",
           dedisp_get_error_string(derror));
    return -1;
  }
  // 执行去分散操作，将输入的滤波信号转换为去分散后的信号
  // dedisp_error derror = dedisp_execute_adv(pl->dedispersion_plan, nsamps,
  //                                          pl->h_clean_filterbank.data(), //
  //                                          输入滤波后的原始信号 nbits, //
  //                                          输入信号的每个样本位宽
  //                                          pl->params.nchans * nbits / 8, //
  //                                          输入信号的跨度（每个时间点的总字节数）
  //                                          pl->h_dm_series.data(),        //
  //                                          输出去分散信号存储位置
  //                                          pl->params.dm_nbits,           //
  //                                          输出信号的每个样本位宽
  //                                          series_stride *
  //                                          pl->params.dm_nbits / 8, //
  //                                          输出信号的跨度 0); // 执行参数标志
  stop_timer(dedisp_timer);
  if (derror != DEDISP_NO_ERROR) {
    return throw_dedisp_error(derror);
  }

  if (beam == 0 && first_idx == 0) {
    // TESTING
    // write_host_time_series((unsigned int*)out, nsamps_computed, out_nbits,
    //                       pl->params.dt, "dedispersed_0.tim");
  }

  // TESTING
  // hd_size write_dm = 0;

  bool too_many_giants = false;

  // 锁页内存分配
  // dedisp_byte *pinned_dm_series;
  // CHECK(cudaHostAlloc(&pinned_dm_series, pl->h_dm_series.size() *
  // sizeof(hd_byte), cudaHostAllocDefault)); printf("h_dm_series.size = %d,
  // type:%s\n", pl->h_dm_series.size(), typeid(hd_byte).name());

  // 将原始数据移入锁页内存
  // std::memcpy(pinned_dm_series,
  // thrust::raw_pointer_cast(pl->h_dm_series.data()), pl->h_dm_series.size() *
  // sizeof(hd_byte));

  // 配置 CUDA 流数量和 OpenMP 线程数量

  // std::vector<cudaStream_t> streams(N_threads);
  // for (int i = 0; i < N_threads; ++i)
  // {
  //   CHECK(cudaStreamCreate(&streams[i]));
  // }
  // #pragma omp parallel

  // CHECK(cudaMemcpy(d_dm_series,
  // thrust::raw_pointer_cast(pl->h_dm_series.data()), pl->h_dm_series.size() *
  // sizeof(hd_byte), cudaMemcpyHostToDevice));
  start_timer(DSPT_timer);
  omp_set_num_threads(N_threads);
  // printf("N_threads = %d\n", N_threads);
  // omp_set_nested(1);
#pragma omp parallel
  {
    CHECK(cudaSetDevice(pl->params.gpu_id));
    // unsigned int num_threads = omp_get_num_threads();
    // cout<<"num_threads = "<<num_threads<<endl;
    int tid = omp_get_thread_num();
    // cudaStream_t stream = streams[tid];
    hd_size chunk_size = dm_count / N_threads;
    hd_size start_idx = tid * chunk_size;
    hd_size end_idx = (tid == N_threads - 1)
                          ? dm_count
                          : (tid + 1) * chunk_size; // 最后一个线程特殊处理

    for (hd_size dm_idx = start_idx; dm_idx < end_idx; ++dm_idx) {
      hd_size cur_dm_scrunch = scrunch_factors[dm_idx];
      hd_size cur_nsamps = nsamps_computed / cur_dm_scrunch;
      hd_float cur_dt = pl->params.dt * cur_dm_scrunch;

      if (pl->params.verbosity >= 3) {
        cout << "dm_idx     = " << dm_idx << endl;
        cout << "scrunch    = " << scrunch_factors[dm_idx] << endl;
        cout << "cur_nsamps = " << cur_nsamps << endl;
        cout << "dt0        = " << pl->params.dt << endl;
        cout << "cur_dt     = " << cur_dt << endl;
        cout << "\tBaselining and normalising each beam..." << endl;
      }
      // 每个线程使用不同的缓冲区位置

      hd_float *time_series =
          thrust::raw_pointer_cast(pl->d_time_series[tid].data());

      hd_size offset = dm_idx * series_stride * pl->params.dm_nbits / 8;

      switch (pl->params.dm_nbits) {
      case 8:
        CHECK(cudaMemcpy(
            time_series, static_cast<unsigned char *>(&d_dm_series[offset]),
            cur_nsamps * sizeof(dedisp_byte), cudaMemcpyDeviceToDevice));
        break;
      case 16:
        CHECK(cudaMemcpy(time_series, (unsigned short *)(&d_dm_series[offset]),
                         cur_nsamps * sizeof(unsigned short),
                         cudaMemcpyDeviceToDevice));
        break;
      case 32:
        CHECK(cudaMemcpy(time_series, (float *)(&d_dm_series[offset]),
                         cur_nsamps * sizeof(float), cudaMemcpyDeviceToDevice));
        break;
      default:
        break;
        // return HD_INVALID_NBITS;
      }
      // CHECK(cudaStreamSynchronize(stream));
      if (tid == 0)
        start_timer(baseline_timer);

      // 后续处理：基线移除、归一化、匹配滤波和巨脉冲检测
      hd_size nsamps_smooth =
          hd_size(pl->params.baseline_length / (2 * cur_dt));

      // // #pragma omp critical
      error =
          baseline_removers[tid].exec(time_series, cur_nsamps, nsamps_smooth);
      if (tid == 0)
        stop_timer(baseline_timer);

      if (pl->params.verbosity >= 2)
        printf("baseline_remover\n");
      // if (error != HD_NO_ERROR)
      // {
      //   return throw_error(error);
      // }

      // #pragma omp critical
      //       {
      if (tid == 0)
        start_timer(normalise_timer);
      hd_float rms = rms_getters[tid].exec(time_series, cur_nsamps);
      thrust::transform(
          pl->d_time_series[tid].begin(), pl->d_time_series[tid].end(),
          thrust::make_constant_iterator(1.0 / rms),
          pl->d_time_series[tid].begin(), thrust::multiplies<hd_float>());

      // baseline_normalizer[tid].exec(time_series, cur_nsamps, nsamps_smooth);
      if (tid == 0)
        stop_timer(normalise_timer);

      // }
      // CHECK(cudaStreamSynchronize(stream));
      // 合并函数
      // baseline_normalizer[tid].exec(time_series, cur_nsamps, nsamps_smooth);

      // 匹配滤波操作

      hd_size rel_boxcar_max = pl->params.boxcar_max / cur_dm_scrunch;
      hd_size max_nsamps_filtered = cur_nsamps + 1 - rel_boxcar_max;
      hd_size cur_filtered_offset = rel_boxcar_max / 2;

      // #pragma omp critical
      //       {

      matched_filter_plans[tid].prep(time_series, cur_nsamps, rel_boxcar_max);

      hd_float *filtered_series =
          thrust::raw_pointer_cast(pl->d_filtered_series[tid].data());

      if (pl->params.verbosity >= 2)
        printf("matched_filter_plan.prep\n");

      // for (hd_size filter_width = cur_dm_scrunch; filter_width <=
      // pl->params.boxcar_max; filter_width *= 2)
      // {
      //   hd_size rel_filter_width = filter_width / cur_dm_scrunch;
      //   hd_size rel_tscrunch_width = std::max(2 * rel_filter_width /
      //   std::max(pl->params.min_tscrunch_width / cur_dm_scrunch, hd_size(1)),
      //   hd_size(1));

      //   matched_filter_plan.exec(filtered_series, rel_filter_width,
      //   rel_tscrunch_width);

      //   hd_size cur_nsamps_filtered = (max_nsamps_filtered - 1) /
      //   rel_tscrunch_width + 1;
      //   thrust::transform(thrust::cuda::par.on(stream),
      //                     thrust::device_pointer_cast(filtered_series),
      //                     thrust::device_pointer_cast(filtered_series +
      //                     cur_nsamps_filtered),
      //                     thrust::make_constant_iterator(1.0 /
      //                     sqrt(static_cast<hd_float>(rel_filter_width))),
      //                     thrust::device_pointer_cast(filtered_series),
      //                     thrust::multiplies<hd_float>());
      // }

      // For each boxcar filter
      // Note: We cannot detect pulse widths < current time resolution
      for (hd_size filter_width = cur_dm_scrunch;
           filter_width <= pl->params.boxcar_max; filter_width *= 2) {
        hd_size rel_filter_width = filter_width / cur_dm_scrunch;
        hd_size filter_idx = get_filter_index(filter_width);

        // Note: Filter width is relative to the current time resolution
        hd_size rel_min_tscrunch_width = std::max(
            pl->params.min_tscrunch_width / cur_dm_scrunch, hd_size(1));
        hd_size rel_tscrunch_width =
            std::max(2 * rel_filter_width / rel_min_tscrunch_width, hd_size(1));
        // Filter width relative to cur_dm_scrunch AND tscrunch
        hd_size rel_rel_filter_width = rel_filter_width / rel_tscrunch_width;
        if (tid == 0)
          start_timer(filter_timer);

        error = matched_filter_plans[tid].exec(
            filtered_series, rel_filter_width, rel_tscrunch_width);

        if (pl->params.verbosity >= 2)
          printf("matched_filter_plan.exec\n");

        // if (error != HD_NO_ERROR)
        // {
        //   return throw_error(error);
        // }
        // Divide and round up
        hd_size cur_nsamps_filtered =
            ((max_nsamps_filtered - 1) / rel_tscrunch_width + 1);
        hd_size cur_scrunch = cur_dm_scrunch * rel_tscrunch_width;

        if (pl->params.boxcar_renorm) {
          // recompute then RMS of the filtered time series, then use that for
          // rescaling. Note that this method reduces the S/N of injected
          // pulses. For more information see
          // https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.2316G/abstract
          // [Appendix A]

          hd_float rms =
              rms_getters[tid].exec(filtered_series, cur_nsamps_filtered);

          thrust::transform(thrust::device_ptr<hd_float>(filtered_series),
                            thrust::device_ptr<hd_float>(filtered_series) +
                                cur_nsamps_filtered,
                            thrust::make_constant_iterator(hd_float(1.0) / rms),
                            thrust::device_ptr<hd_float>(filtered_series),
                            thrust::multiplies<hd_float>());
        } else {
          // rescale the filtered time series (RMS ~ sqrt(time))

          thrust::constant_iterator<hd_float> norm_val_iter(
              1.0 / sqrt((hd_float)rel_filter_width));

          thrust::transform(thrust::device_ptr<hd_float>(filtered_series),
                            thrust::device_ptr<hd_float>(filtered_series) +
                                cur_nsamps_filtered,
                            norm_val_iter,
                            thrust::device_ptr<hd_float>(filtered_series),
                            thrust::multiplies<hd_float>());
        }
        if (tid == 0)
          stop_timer(filter_timer);
        hd_size prev_giant_count = d_giant_peaks_t[tid].size();

        if (tid == 0)
          start_timer(giants_timer);

        // #pragma omp critical
        error = giant_finders[tid].exec(
            filtered_series, cur_nsamps_filtered, pl->params.detect_thresh,
            // pl->params.cand_sep_time,
            //  Note: This was MB's recommendation
            pl->params.cand_sep_time * rel_rel_filter_width,
            d_giant_peaks_t[tid], d_giant_inds_t[tid], d_giant_begins_t[tid],
            d_giant_ends_t[tid]);

        if (pl->params.verbosity >= 2)
          printf("giant_finder.exec\n");
        //           thrust::host_vector<hd_float> h_giant_inds =
        //           d_giant_inds_t[tid];  // 将设备向量拷贝到主机
        // for (int i = 0; i < h_giant_inds.size(); ++i) {
        //     std::cout << "h_giant_inds[" << i << "] = " << h_giant_inds[i] <<
        //     std::endl;
        // }

        // if (error != HD_NO_ERROR)
        // {
        //   return throw_error(error);
        // }

        hd_size rel_cur_filtered_offset =
            (cur_filtered_offset / rel_tscrunch_width);

        using namespace thrust::placeholders;

        thrust::transform(d_giant_inds_t[tid].begin() + prev_giant_count,
                          d_giant_inds_t[tid].end(),
                          d_giant_inds_t[tid].begin() + prev_giant_count,
                          /*first_idx +*/ (_1 + rel_cur_filtered_offset) *
                              cur_scrunch);

        thrust::transform(d_giant_begins_t[tid].begin() + prev_giant_count,
                          d_giant_begins_t[tid].end(),
                          d_giant_begins_t[tid].begin() + prev_giant_count,
                          /*first_idx +*/ (_1 + rel_cur_filtered_offset) *
                              cur_scrunch);
        thrust::transform(d_giant_ends_t[tid].begin() + prev_giant_count,
                          d_giant_ends_t[tid].end(),
                          d_giant_ends_t[tid].begin() + prev_giant_count,
                          /*first_idx +*/ (_1 + rel_cur_filtered_offset) *
                              cur_scrunch);
        // CHECK(cudaStreamSynchronize(stream));

        // #pragma omp critical
        // {
        d_giant_filter_inds_t[tid].resize(d_giant_peaks_t[tid].size(),
                                          filter_idx);
        d_giant_dm_inds_t[tid].resize(d_giant_peaks_t[tid].size(), dm_idx);
        // Note: This could be used to track total member samples if desired
        d_giant_members_t[tid].resize(d_giant_peaks_t[tid].size(), 1);

        // }
        if (tid == 0)
          stop_timer(giants_timer);
      } // End of filter width loop
      // }
      // printf("end of filter loop\n");
    } // DMs for each thread
    // CHECK(cudaStreamSynchronize(stream));
  }
  cudaDeviceSynchronize();
#pragma omp barrier
  // // 释放基线移除和归一化计划
  // for (int i = 0; i < N_threads; ++i)
  // {
  //   baseline_normalizer[i].cleanup();
  // }
  // cout << "fuse_timer = " << baseline_timer.getTime() << endl;

  // printf("thread complete!\n");
  stop_timer(DSPT_timer);

  // for (size_t i = 0; i < N_threads; ++i)
  // {
  //   cout << "thread " << i << " grint indx size is:" <<
  //   d_giant_inds_t[i].size() << endl;
  // }

  // size_t total_size = 0;
  // for (size_t i = 0; i < N_threads; ++i)
  // {
  //   total_size += d_giant_peaks_t[i].size();
  // }
  thrust::device_vector<hd_float> d_giant_peaks;
  thrust::device_vector<hd_size> d_giant_inds;
  thrust::device_vector<hd_size> d_giant_begins;
  thrust::device_vector<hd_size> d_giant_ends;
  thrust::device_vector<hd_size> d_giant_filter_inds;
  thrust::device_vector<hd_size> d_giant_dm_inds;
  thrust::device_vector<hd_size> d_giant_members;

  // 合并 d_giant_
  for (size_t i = 0; i < N_threads; ++i) {
    d_giant_peaks.insert(d_giant_peaks.end(), d_giant_peaks_t[i].begin(),
                         d_giant_peaks_t[i].end());
    d_giant_inds.insert(d_giant_inds.end(), d_giant_inds_t[i].begin(),
                        d_giant_inds_t[i].end());
    d_giant_begins.insert(d_giant_begins.end(), d_giant_begins_t[i].begin(),
                          d_giant_begins_t[i].end());
    d_giant_ends.insert(d_giant_ends.end(), d_giant_ends_t[i].begin(),
                        d_giant_ends_t[i].end());
    d_giant_filter_inds.insert(d_giant_filter_inds.end(),
                               d_giant_filter_inds_t[i].begin(),
                               d_giant_filter_inds_t[i].end());
    d_giant_dm_inds.insert(d_giant_dm_inds.end(), d_giant_dm_inds_t[i].begin(),
                           d_giant_dm_inds_t[i].end());
    d_giant_members.insert(d_giant_members.end(), d_giant_members_t[i].begin(),
                           d_giant_members_t[i].end());
  }
  // cout << "d_giant_peaks size is:" << d_giant_peaks.size() << endl;
  // 清理 CUDA 流和锁页内存
  // for (auto &stream : streams)
  // {
  //   CHECK(cudaStreamDestroy(stream));
  // }
  // CHECK(cudaFreeHost(pinned_dm_series));
  // 注意：不再在这里释放d_dm_series，因为它现在是持久化的
  // 释放内存
  // cudaFree(d_time_series);
  // cudaFree(d_filtered_series);
  // cudaFree(d_dm_series); // 不再在这里释放，由pipeline管理
  hd_size giant_count = d_giant_peaks.size();
  if (pl->params.verbosity >= 2) {
    cout << "Giant count = " << giant_count << endl;
  }

  start_timer(candidates_timer);

  thrust::host_vector<hd_float> h_group_peaks;
  thrust::host_vector<hd_size> h_group_inds;
  thrust::host_vector<hd_size> h_group_begins;
  thrust::host_vector<hd_size> h_group_ends;
  thrust::host_vector<hd_size> h_group_filter_inds;
  thrust::host_vector<hd_size> h_group_dm_inds;
  thrust::host_vector<hd_size> h_group_members;
  thrust::host_vector<hd_float> h_group_dms;

  // if (!too_many_giants)
  //{
  thrust::device_vector<hd_size> d_giant_labels(giant_count);
  hd_size *d_giant_labels_ptr = thrust::raw_pointer_cast(&d_giant_labels[0]);

  RawCandidates d_giants;
  d_giants.peaks = thrust::raw_pointer_cast(&d_giant_peaks[0]);
  d_giants.inds = thrust::raw_pointer_cast(&d_giant_inds[0]);
  d_giants.begins = thrust::raw_pointer_cast(&d_giant_begins[0]);
  d_giants.ends = thrust::raw_pointer_cast(&d_giant_ends[0]);
  d_giants.filter_inds = thrust::raw_pointer_cast(&d_giant_filter_inds[0]);
  d_giants.dm_inds = thrust::raw_pointer_cast(&d_giant_dm_inds[0]);
  d_giants.members = thrust::raw_pointer_cast(&d_giant_members[0]);

  hd_size filter_count = get_filter_index(pl->params.boxcar_max) + 1;

  if (pl->params.verbosity >= 2) {
    cout << "Grouping coincident candidates..." << endl;
  }

  ConstRawCandidates *const_d_giants = (ConstRawCandidates *)&d_giants;

  hd_size label_count;
  // Stopwatch origin_timer;
  // origin_timer.start();
  // error = label_candidate_clusters(
  //     giant_count, *const_d_giants, pl->params.cand_sep_time,
  //     pl->params.cand_sep_filter, pl->params.cand_sep_dm, d_giant_labels_ptr,
  //     &label_count);
  // cout << "origin label_count = " << label_count << endl;
  // origin_timer.stop();
  // cout << "origin label_timer = " << origin_timer.getTime() << endl;

  Stopwatch hash_timer;
  hash_timer.start();
  error = label_candidate_clusters_optimized(
      giant_count,
      // error = label_candidate_clusters_kd_dbscan(giant_count,
      *const_d_giants, pl->params.cand_sep_time, pl->params.cand_sep_filter,
      pl->params.cand_sep_dm, d_giant_labels_ptr, &label_count);
  // cout << "optimized label_count = " << label_count << endl;
  hash_timer.stop();
  // cout << "optimized label_timer = " << hash_timer.getTime() << endl;

  if (error != HD_NO_ERROR) {
    return throw_error(error);
  }

  hd_size group_count = label_count;
  if (pl->params.verbosity >= 2) {
    cout << "Candidate count = " << group_count << endl;
  }

  thrust::device_vector<hd_float> d_group_peaks(group_count);
  thrust::device_vector<hd_size> d_group_inds(group_count);
  thrust::device_vector<hd_size> d_group_begins(group_count);
  thrust::device_vector<hd_size> d_group_ends(group_count);
  thrust::device_vector<hd_size> d_group_filter_inds(group_count);
  thrust::device_vector<hd_size> d_group_dm_inds(group_count);
  thrust::device_vector<hd_size> d_group_members(group_count);

  thrust::device_vector<hd_float> d_group_dms(group_count);

  RawCandidates d_groups;
  d_groups.peaks = thrust::raw_pointer_cast(&d_group_peaks[0]);
  d_groups.inds = thrust::raw_pointer_cast(&d_group_inds[0]);
  d_groups.begins = thrust::raw_pointer_cast(&d_group_begins[0]);
  d_groups.ends = thrust::raw_pointer_cast(&d_group_ends[0]);
  d_groups.filter_inds = thrust::raw_pointer_cast(&d_group_filter_inds[0]);
  d_groups.dm_inds = thrust::raw_pointer_cast(&d_group_dm_inds[0]);
  d_groups.members = thrust::raw_pointer_cast(&d_group_members[0]);

  merge_candidates(giant_count, d_giant_labels_ptr, *const_d_giants, d_groups);

  // Look up the actual DM of each group
  thrust::device_vector<hd_float> d_dm_list(dm_list, dm_list + dm_count);
  thrust::gather(d_group_dm_inds.begin(), d_group_dm_inds.end(),
                 d_dm_list.begin(), d_group_dms.begin());

  // Device to host transfer of candidates
  h_group_peaks = d_group_peaks;
  h_group_inds = d_group_inds;
  h_group_begins = d_group_begins;
  h_group_ends = d_group_ends;
  h_group_filter_inds = d_group_filter_inds;
  h_group_dm_inds = d_group_dm_inds;
  h_group_members = d_group_members;
  h_group_dms = d_group_dms;
  // h_group_flags = d_group_flags;
  //}

  if (pl->params.verbosity >= 2) {
    cout << "Writing output candidates, utc_start=" << pl->params.utc_start
         << endl;
  }

  char buffer[64];
  time_t now = pl->params.utc_start +
               (time_t)(first_idx / pl->params.spectra_per_second);
  strftime(buffer, 64, HD_TIMESTR, (struct tm *)gmtime(&now));

  std::stringstream ss;
  ss << std::setw(2) << std::setfill('0') << pl->params.beam + 1;

  std::ostringstream oss;

  if (pl->params.coincidencer_host != NULL &&
      pl->params.coincidencer_port != -1) {
    try {
      ClientSocket client_socket(pl->params.coincidencer_host,
                                 pl->params.coincidencer_port);

      strftime(buffer, 64, HD_TIMESTR,
               (struct tm *)gmtime(&(pl->params.utc_start)));

      oss << buffer << " ";

      time_t now = pl->params.utc_start +
                   (time_t)(first_idx / pl->params.spectra_per_second);
      strftime(buffer, 64, HD_TIMESTR, (struct tm *)gmtime(&now));
      oss << buffer << " ";

      oss << first_idx << " ";
      oss << ss.str() << " ";
      oss << h_group_peaks.size() << endl;
      client_socket << oss.str();
      oss.flush();
      oss.str("");

      for (hd_size i = 0; i < h_group_peaks.size(); ++i) {
        hd_size samp_idx = first_idx + h_group_inds[i];
        oss << h_group_peaks[i] << "\t" << samp_idx << "\t"
            << samp_idx * pl->params.dt << "\t" << h_group_filter_inds[i]
            << "\t" << h_group_dm_inds[i] << "\t" << h_group_dms[i] << "\t"
            << h_group_members[i] << "\t" << first_idx + h_group_begins[i]
            << "\t" << first_idx + h_group_ends[i] << endl;

        client_socket << oss.str();
        oss.flush();
        oss.str("");
      }
      // client_socket should close when it goes out of scope...
    } catch (SocketException &e) {
      std::cerr << "SocketException was caught:" << e.description() << "\n";
    }
  } else {
    if (pl->params.verbosity >= 2)
      cout << "Output timestamp: " << buffer << endl;
    // get sigproc filename
    // std::string sig_file = std::string(pl->params.sigproc_file);
    // sig_file = sig_file.substr(sig_file.find_last_of('/') + 1);
    // sig_file = sig_file.substr(0, sig_file.length() - 4);
    // cout<<"sig_file: "<<sig_file<<endl;
    // std::string filename = std::string(pl->params.output_dir) + "/" +
    // sig_file + ".cand";

    std::string filename = std::string(pl->params.output_dir) + "/" +
                           std::string(buffer) + "_" + ss.str() + ".cand";

    if (pl->params.verbosity >= 2)
      cout << "Output filename: " << filename << endl;

    std::ofstream cand_file(filename.c_str(), std::ios::out);
    if (pl->params.verbosity >= 2)
      cout << "Dumping " << h_group_peaks.size() << " candidates to "
           << filename << endl;

    if (cand_file.good()) {
      for (hd_size i = 0; i < h_group_peaks.size(); ++i) {
        hd_size samp_idx = first_idx + h_group_inds[i];
        cand_file << h_group_peaks[i] << "\t" << samp_idx << "\t"
                  << samp_idx * pl->params.dt << "\t" << h_group_filter_inds[i]
                  << "\t" << h_group_dm_inds[i] << "\t" << h_group_dms[i]
                  << "\t" << h_group_members[i] << "\t"
                  << first_idx + h_group_begins[i] << "\t"
                  << first_idx + h_group_ends[i] << "\t"
                  << "\n";
      }
    } else
      cout << "Skipping dump due to bad file open on " << filename << endl;
    cand_file.close();
  }

  stop_timer(candidates_timer);

  stop_timer(total_timer);

#ifdef HD_BENCHMARK
  if (pl->params.verbosity >= 1) {
    cout << "Mem alloc time:          " << memory_timer.getTime() << endl;
    cout << "0-DM cleaning time:      " << clean_timer.getTime() << endl;
    cout << "Dedispersion time:       " << dedisp_timer.getTime() << endl;
    cout << "Copy time:               " << copy_timer.getTime() << endl;
    cout << "Baselining time:         " << baseline_timer.getTime() << endl;
    cout << "Normalisation time:      " << normalise_timer.getTime() << endl;
    cout << "Filtering time:          " << filter_timer.getTime() << endl;
    cout << "Find giants time:        " << giants_timer.getTime() << endl;
    cout << "Process candidates time: " << candidates_timer.getTime() << endl;
    cout << "DSPT time:               " << DSPT_timer.getTime() << endl;
    cout << "Total time:              " << total_timer.getTime() << endl;
  }
  // cached_allocator::get_instance().print_detailed_report();
  // cached_allocator::get_instance().print_statistics();
  // cached_allocator::get_instance().print_summary();

  hd_float time_sum =
      (memory_timer.getTime() + clean_timer.getTime() + dedisp_timer.getTime() +
       copy_timer.getTime() + baseline_timer.getTime() +
       normalise_timer.getTime() + filter_timer.getTime() +
       giants_timer.getTime() + candidates_timer.getTime());
  hd_float misc_time = total_timer.getTime() - time_sum;

  /*
  std::ofstream timing_file("timing.dat", std::ios::app);
  timing_file << total_timer.getTime() << "\t"
              << misc_time << "\t"
              << memory_timer.getTime() << "\t"
              << clean_timer.getTime() << "\t"
              << dedisp_timer.getTime() << "\t"
              << copy_timer.getTime() << "\t"
              << baseline_timer.getTime() << "\t"
              << normalise_timer.getTime() << "\t"
              << filter_timer.getTime() << "\t"
              << giants_timer.getTime() << "\t"
              << candidates_timer.getTime() << endl;
  timing_file.close();
  */

#endif // HD_BENCHMARK

  if (too_many_giants) {
    return HD_TOO_MANY_EVENTS;
  } else {
    return HD_NO_ERROR;
  }
}

void hd_destroy_pipeline(hd_pipeline pipeline) {
  if (pipeline->params.verbosity >= 2) {
    cout << "\tDeleting pipeline object..." << endl;
  }

  dedisp_destroy_plan(pipeline->dedispersion_plan);

  // 释放持久化的DM系列缓冲区
  if (pipeline->d_dm_series != nullptr) {
    cudaFree(pipeline->d_dm_series);
    pipeline->d_dm_series = nullptr;
    pipeline->d_dm_series_allocated_size = 0;
  }

  // Note: This assumes memory owned by pipeline cleans itself up
  if (pipeline) {
    delete pipeline;
  }
}
