#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hd/default_params.h"
#include "hd/error.h"
#include "hd/parse_command_line.h"
#include "hd/pipeline.h"

// input formats supported
#include "hd/DataSource.h"
#include "hd/SigprocFile.h"
#ifdef HAVE_PSRDADA
#include "hd/PSRDadaRingBuffer.h"
#endif

#include "hd/stopwatch.h"
#include <chrono>
#include <omp.h>

// 定义表示Pipeline任务的结构体
struct PipelineTask {
  hd_pipeline pipeline;            // 已创建的pipeline
  std::vector<hd_byte> filterbank; // 文件数据缓冲区
  std::string filename;            // 文件名
  size_t nsamps_gulp;              // 一次读取的样本数
  size_t stride;                   // 步长
  size_t nbits;                    // 位深度
  DataSource *data_source;         // 数据源

  // 构造函数和析构函数
  PipelineTask() : pipeline(nullptr), data_source(nullptr) {}
  ~PipelineTask() {
    if (pipeline) {
      hd_destroy_pipeline(pipeline);
    }
    if (data_source) {
      delete data_source;
    }
  }
};

// 线程安全的任务队列
class TaskQueue {
private:
  std::queue<std::shared_ptr<PipelineTask>> tasks;
  std::mutex mutex;
  std::condition_variable condition;
  std::condition_variable not_full_condition; // 队列未满条件变量
  std::atomic<bool> stop_flag{false};
  size_t max_size; // 最大队列容量

public:
  // 构造函数，设置最大容量
  TaskQueue(size_t capacity = SIZE_MAX) : max_size(capacity) {}

  // 添加任务到队列
  void push(std::shared_ptr<PipelineTask> task) {
    std::unique_lock<std::mutex> lock(mutex);
    // 等待队列有空间
    while (tasks.size() >= max_size && !stop_flag) {
      not_full_condition.wait(lock);
    }
    if (stop_flag)
      return; // 如果已停止，不添加新任务

    tasks.push(task);
    lock.unlock();
    condition.notify_one();
  }

  // 从队列获取任务
  std::shared_ptr<PipelineTask> pop() {
    std::unique_lock<std::mutex> lock(mutex);
    // 当队列为空且未停止时等待
    while (tasks.empty() && !stop_flag) {
      condition.wait(lock);
    }
    // 如果队列为空且已停止，返回nullptr
    if (tasks.empty()) {
      return nullptr;
    }
    std::shared_ptr<PipelineTask> task = tasks.front();
    tasks.pop();
    lock.unlock();
    not_full_condition.notify_one(); // 通知队列有空间了
    return task;
  }

  // 停止所有等待线程
  void stop() {
    std::unique_lock<std::mutex> lock(mutex);
    stop_flag = true;
    lock.unlock();
    condition.notify_all();
    not_full_condition.notify_all(); // 通知所有等待的push操作
  }

  bool empty() {
    std::unique_lock<std::mutex> lock(mutex);
    return tasks.empty();
  }

  size_t size() {
    std::unique_lock<std::mutex> lock(mutex);
    return tasks.size();
  }
};

// Create pipeline function
void create_pipeline_worker(TaskQueue &create_queue, TaskQueue &execute_queue,
                            const hd_params &params) {
  while (true) {
    // Get task from create queue
    std::shared_ptr<PipelineTask> task = create_queue.pop();
    if (!task)
      break; // Queue stopped

    // Set parameters
    hd_params pipeline_params = params;
    pipeline_params.sigproc_file = task->filename.c_str();

    // Create data source
    task->data_source =
        new SigprocFile(pipeline_params.sigproc_file, pipeline_params.fswap);
    if (!task->data_source || task->data_source->get_error()) {
      cerr << "ERROR: Failed to open data file: " << task->filename << endl;
      continue;
    }

    // Configure beam parameters
    if (!pipeline_params.override_beam) {
      if (task->data_source->get_beam() > 0)
        pipeline_params.beam = task->data_source->get_beam() - 1;
      else
        pipeline_params.beam = 0;
    }

    pipeline_params.f0 = task->data_source->get_f0();
    pipeline_params.df = task->data_source->get_df();
    pipeline_params.dt = task->data_source->get_tsamp();
    pipeline_params.nchans = task->data_source->get_nchan();
    pipeline_params.utc_start = task->data_source->get_utc_start();
    pipeline_params.spectra_per_second = task->data_source->get_spectra_rate();

    // Get file parameters
    task->stride = task->data_source->get_stride();
    task->nbits = task->data_source->get_nbit();
    task->nsamps_gulp = pipeline_params.nsamps_gulp;

    // Check channel count is multiple of 16
    if (pipeline_params.nchans % 16 != 0) {
      cerr << "ERROR: Dedisp library supports multiples of 16 channels only: "
           << task->filename << endl;
      continue;
    }

    // Allocate buffer
    size_t filterbank_bytes = 2 * task->nsamps_gulp * task->stride;
    task->filterbank.resize(filterbank_bytes);

    // Create pipeline
    hd_error error = hd_create_pipeline(&task->pipeline, pipeline_params);
    if (error != HD_NO_ERROR) {
      cerr << "ERROR: Pipeline creation failed for file: " << task->filename
           << endl;
      cerr << "       " << hd_get_error_string(error) << endl;
      continue;
    }

    // Add task to execute queue
    execute_queue.push(task);
  }
}

// Execute pipeline function
void execute_pipeline_worker(TaskQueue &execute_queue,
                             const hd_params &params) {
  while (true) {
    // Get task from execute queue
    std::shared_ptr<PipelineTask> task = execute_queue.pop();
    if (!task)
      break; // Queue stopped

    // Execute pipeline
    size_t total_nsamps = 0;
    size_t nsamps_read = task->data_source->get_data(
        task->nsamps_gulp, (char *)&task->filterbank[0]);
    size_t overlap = 0;
    bool stop_requested = false;

    while (nsamps_read && !stop_requested) {

      hd_size nsamps_processed;
      hd_error error = hd_execute(task->pipeline, &task->filterbank[0],
                                  nsamps_read + overlap, task->nbits,
                                  total_nsamps, &nsamps_processed);

      if (error == HD_NO_ERROR) {
        // Success - continue processing
      } else if (error == HD_TOO_MANY_EVENTS) {
        if (params.verbosity >= 1)
          cerr << "WARNING: Too many events, some data skipped: "
               << task->filename << endl;
      } else {
        cerr << "ERROR: Pipeline execution failed: " << task->filename << endl;
        cerr << "       " << hd_get_error_string(error) << endl;
        break;
      }

      total_nsamps += nsamps_processed;

      // Reposition for unprocessed samples
      std::copy(&task->filterbank[nsamps_processed * task->stride],
                &task->filterbank[(nsamps_read + overlap) * task->stride],
                &task->filterbank[0]);
      overlap += nsamps_read - nsamps_processed;
      nsamps_read = task->data_source->get_data(
          task->nsamps_gulp, (char *)&task->filterbank[overlap * task->stride]);

      // Stop processing when file ends
      if (nsamps_read < task->nsamps_gulp)
        stop_requested = true;
    }

    // Process final partial block
    if (stop_requested && nsamps_read > 0) {

      hd_size nsamps_processed;
      hd_size nsamps_to_process = nsamps_read + overlap;
      if (nsamps_to_process > task->nsamps_gulp)
        nsamps_to_process = task->nsamps_gulp;

      hd_error error =
          hd_execute(task->pipeline, &task->filterbank[0], nsamps_to_process,
                     task->nbits, total_nsamps, &nsamps_processed);

      if (error != HD_NO_ERROR && error != HD_TOO_MANY_EVENTS &&
          error != HD_TOO_FEW_NSAMPS) {
        cerr << "ERROR: Pipeline execution failed: " << task->filename << endl;
        cerr << "       " << hd_get_error_string(error) << endl;
      }

      total_nsamps += nsamps_processed;
    }

    if (params.verbosity >= 1) {
      cout << "Completed: " << task->filename << " (" << total_nsamps
           << " samples)" << endl;
    }
  }
}

// Check if file is .fil format
bool is_filterbank_file(const std::string &path) {
  return path.size() >= 4 && path.substr(path.size() - 4) == ".fil";
}

int main(int argc, char *argv[]) {

  omp_set_nested(1);
  auto start = std::chrono::high_resolution_clock::now();

  // Parse command line arguments
  hd_params params;
  hd_set_default_params(&params);
  int ok = hd_parse_command_line(argc, argv, &params);

  if (ok < 0)
    return 1;

  // Set thread configuration from command line parameters
  const unsigned int num_create_threads = 2;
  const unsigned int num_execute_threads = params.num_execute_threads;

  if (params.verbosity >= 1) {
    cout << "Create threads: " << num_create_threads << endl;
    cout << "Execute threads: " << num_execute_threads << endl;
  }

  // Create task queues
  TaskQueue create_queue(6);
  TaskQueue execute_queue(6);

  // Get file list
  std::vector<std::string> files_to_process;

  // Check if input is file or directory
  if (params.sigproc_file != nullptr) {
    std::filesystem::path path(params.sigproc_file);

    if (std::filesystem::is_directory(path)) {
      // Directory: collect all .fil files
      for (const auto &entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() &&
            is_filterbank_file(entry.path().string())) {
          files_to_process.push_back(entry.path().string());
        }
      }

      if (files_to_process.empty()) {
        cerr << "ERROR: No .fil files found in directory "
             << params.sigproc_file << endl;
        return 1;
      }

      if (1) {
        cout << "Found " << files_to_process.size() << " .fil files in "
             << params.sigproc_file << endl;
      }
    } else if (std::filesystem::is_regular_file(path)) {
      // Single file
      files_to_process.push_back(std::string(params.sigproc_file));
    } else {
      cerr << "ERROR: Path " << params.sigproc_file
           << " is neither file nor directory" << endl;
      return 1;
    }
  } else {
    cerr << "ERROR: No input file or directory specified" << endl;
    hd_print_usage();
    return 1;
  }

  // Create worker threads
  std::vector<std::thread> create_threads;
  std::vector<std::thread> execute_threads;

  // Start create pipeline threads
  for (unsigned int i = 0; i < num_create_threads; i++) {
    create_threads.emplace_back(create_pipeline_worker, std::ref(create_queue),
                                std::ref(execute_queue), std::ref(params));
  }

  // Start execute pipeline threads
  for (unsigned int i = 0; i < num_execute_threads; i++) {
    execute_threads.emplace_back(execute_pipeline_worker,
                                 std::ref(execute_queue), std::ref(params));
  }

  // Add files to create queue
  for (const auto &file : files_to_process) {
    auto task = std::make_shared<PipelineTask>();
    task->filename = file;

    create_queue.push(task);
  }

  // Stop create queue and wait for completion
  create_queue.stop();

  // Wait for all create threads to complete
  for (auto &thread : create_threads) {
    thread.join();
  }

  // Wait for execute queue to empty, then stop
  while (!execute_queue.empty()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  execute_queue.stop();

  // Wait for all execute threads to complete
  for (auto &thread : execute_threads) {
    thread.join();
  }

  if (params.verbosity >= 1) {
    cout << "All files processed" << endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  cout << "Total execution time: " << elapsed.count() << " seconds" << endl;

  return 0;
}