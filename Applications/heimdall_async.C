/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

 #include <iostream>
 using std::cout;
 using std::cerr;
 using std::endl;
 #include <sstream>
 #include <fstream>
 #include <iomanip>
 #include <vector>
 #include <future>
 #include <memory>
 
 #include "hd/parse_command_line.h"
 #include "hd/default_params.h"
 #include "hd/pipeline.h"
 #include "hd/error.h"
 
 // input formats supported
 #include "hd/DataSource.h"
 #include "hd/SigprocFile.h"
 #ifdef HAVE_PSRDADA
 #include "hd/PSRDadaRingBuffer.h"
 #endif
 
 #include "hd/stopwatch.h"
 #include <chrono>
 
 // 用于测量时间的辅助函数
 double get_time_diff(std::chrono::high_resolution_clock::time_point start) {
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> diff = end - start;
   return diff.count();
 }
 
 int main(int argc, char* argv[]) 
 {
   auto start = std::chrono::high_resolution_clock::now();
   auto pre_s = std::chrono::high_resolution_clock::now();
   hd_params params;
   hd_set_default_params(&params);
   int ok = hd_parse_command_line(argc, argv, &params);
   size_t nsamps_gulp = params.nsamps_gulp;
 
   if (ok < 0)
     return 1;
   
   DataSource* data_source = 0;
   // SigprocFile对象指针，用于异步读取
   SigprocFile* sigproc_file = 0;
 
 #ifdef HAVE_PSRDADA
   if( params.dada_id != 0 ) {
 
     if (params.verbosity)
       cerr << "Createing PSRDADA client" << endl;
 
     PSRDadaRingBuffer * d = new PSRDadaRingBuffer(params.dada_id);
 
     // Read from psrdada ring buffer
     if( !d || d->get_error() ) {
       cerr << "ERROR: Failed to initialise connection to psrdada" << endl;
       return -1;
     }
 
     if (params.verbosity)
       cerr << "Connecting to ring buffer" << endl;
     // connect to PSRDADA ring buffer
     if (! d->connect())
     {
        cerr << "ERROR: Failed to connection to psrdada ring buffer" << endl;
       return -1;
     }
 
     if (params.verbosity)
       cerr << "Waiting for next header / data" << endl;
 
     // wait for and then read next PSRDADA header/observation
     if (! d->read_header())
     {
        cerr << "ERROR: Failed to connection to psrdada ring buffer" << endl;
       return -1;
     }
 
     data_source = (DataSource *) d;
     if (!params.override_beam)
       params.beam = d->get_beam() - 1;
   }
   else 
   {
 #endif
     // Read from filterbank file
     sigproc_file = new SigprocFile(params.sigproc_file, params.fswap);
     data_source = sigproc_file;
     if( !data_source || data_source->get_error() ) {
       cerr << "ERROR: Failed to open data file" << endl;
       return -1;
     }
 #ifdef HAVE_PSRDADA
   }
 #endif
 
   if (!params.override_beam)
   {
     if (data_source->get_beam() > 0)
       params.beam = data_source->get_beam() - 1;
     else
       params.beam = 0;
   }
 
   params.f0 = data_source->get_f0();
   params.df = data_source->get_df();
   params.dt = data_source->get_tsamp();
 
   if ( params.verbosity > 0)
     cout << "processing beam " << (params.beam+1)  << endl;
 
   size_t stride = data_source->get_stride();
   size_t nbits  = data_source->get_nbit();
 
   params.nchans = data_source->get_nchan();
   params.utc_start = data_source->get_utc_start();
   params.spectra_per_second = data_source->get_spectra_rate();
 
   // warn about dedisp bug of modulo 16 channels
   if (params.nchans % 16 != 0)
   {
     cerr << "ERROR: Dedisp library supports multiples of 16 channels only" << endl;
     return -1;
   }
 
   // 使用简单的缓冲区策略：处理缓冲区 + 预读缓冲区
   size_t buffer_bytes = 2 * nsamps_gulp * stride; // 为overlap预留足够空间
   if ( params.verbosity >= 2)
     cout << "allocating filterbank data vector for " << nsamps_gulp
          << " samples with size " << buffer_bytes << " bytes" << endl;
   
   // 主处理缓冲区（与原版保持一致）
   std::vector<hd_byte> filterbank(buffer_bytes);
   // 异步预读缓冲区
   std::vector<hd_byte> prefetch_buffer(nsamps_gulp * stride);
   
   bool stop_requested = false;
   auto pre_e = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> pre_time = pre_e - pre_s;
   cout<< "pre time: "<< pre_time.count() << endl;
   
   // Create the pipeline object
   // --------------------------
   auto pip_create_s = std::chrono::high_resolution_clock::now();
   hd_pipeline pipeline;
   hd_error error;
   error = hd_create_pipeline(&pipeline, params);
   if( error != HD_NO_ERROR ) {
     cerr << "ERROR: Pipeline creation failed" << endl;
     cerr << "       " << hd_get_error_string(error) << endl;
     return -1;
   }
   // --------------------------
   
   if( params.verbosity >= 1 ) {
     cout << "Beginning data processing, requesting " << nsamps_gulp << " samples" << endl;
   }
   auto pip_create_e = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> pip_create_time = pip_create_e - pip_create_s;
   cout<< "pipeline create time: "<< pip_create_time.count() << endl;
 
 
   // start a timer for the whole pipeline
   auto pipeline_s = std::chrono::high_resolution_clock::now();
   
   // 添加IO和处理时间统计
   double total_io_time = 0.0;
   double total_processing_time = 0.0;
   double total_wait_time = 0.0;
   int iteration_count = 0;
 
   size_t total_nsamps = 0;
   size_t overlap = 0;
   std::future<size_t> read_future;
   
   // 初始读取到主缓冲区
   auto io_start = std::chrono::high_resolution_clock::now();
   size_t nsamps_read = data_source->get_data(nsamps_gulp, (char*)&filterbank[0]);
   total_io_time += get_time_diff(io_start);
   
   // 立即开始异步预读下一批数据
   if (sigproc_file && nsamps_read > 0) {
     read_future = sigproc_file->get_data_async(nsamps_gulp, (char*)&prefetch_buffer[0]);
   }
   
   while (nsamps_read && !stop_requested)
   {
     iteration_count++;
     
     if (params.verbosity >= 1) {
       cout << "Executing pipeline on new gulp of " << nsamps_read
            << " samples..." << endl;
     }
 
     if (params.verbosity >= 2) {
       cout << " nsamp_gulp=" << nsamps_gulp << " overlap=" << overlap
            << " nsamps_read=" << nsamps_read << " nsamps_read+overlap="
            << nsamps_read+overlap << endl;
     }
       
     // 边界检查
     if ((nsamps_read + overlap) * stride > filterbank.size()) {
       cerr << "ERROR: Buffer overflow detected!" << endl;
       cerr << "  Required bytes: " << (nsamps_read + overlap) * stride << endl;
       cerr << "  Buffer size: " << filterbank.size() << endl;
       break;
     }
     
     // 测量处理时间
     auto process_start = std::chrono::high_resolution_clock::now();
     hd_size nsamps_processed;
     error = hd_execute(pipeline, &filterbank[0], nsamps_read+overlap, nbits,
                        total_nsamps, &nsamps_processed);
     total_processing_time += get_time_diff(process_start);
                        
     if (error == HD_NO_ERROR)
     {
       if (params.verbosity >= 1)
         cout << "Processed " << nsamps_processed << " samples." << endl;
     }
     else if (error == HD_TOO_MANY_EVENTS) 
     {
       if (params.verbosity >= 1)
         cerr << "WARNING: hd_execute produces too many events, some data skipped" << endl;
     }
     else 
     {
       cerr << "ERROR: Pipeline execution failed" << endl;
       cerr << "       " << hd_get_error_string(error) << endl;
       hd_destroy_pipeline(pipeline);
       return -1;
     }
 
     if (params.verbosity >= 1)
       cout << "Main: nsamps_processed=" << nsamps_processed << endl;
 
     total_nsamps += nsamps_processed;
     
     // 处理重叠部分：移动未处理的数据到缓冲区开头（与原版一致）
     std::copy(&filterbank[nsamps_processed * stride],
               &filterbank[(nsamps_read+overlap) * stride],
               &filterbank[0]);
     overlap += nsamps_read - nsamps_processed;
     
     // 等待异步读取完成
     auto wait_start = std::chrono::high_resolution_clock::now();
     if (sigproc_file && read_future.valid()) {
       nsamps_read = read_future.get();
     } else {
       nsamps_read = 0;
     }
     double wait_time = get_time_diff(wait_start);
     total_wait_time += wait_time;
     
     // 将预读数据复制到主缓冲区的正确位置
     if (nsamps_read > 0) {
       std::copy(&prefetch_buffer[0],
                 &prefetch_buffer[nsamps_read * stride],
                 &filterbank[overlap * stride]);
       
       // 开始异步读取下一批数据
       if (sigproc_file) {
         read_future = sigproc_file->get_data_async(nsamps_gulp, (char*)&prefetch_buffer[0]);
       }
     } else {
       stop_requested = true;
     }
     
     // 如果读取的数据小于请求的数据量，说明接近文件末尾
     if (nsamps_read < nsamps_gulp) {
       stop_requested = true;
     }
     
     // 输出每次迭代的时间统计
     if (params.verbosity >= 1) {
       cout << "Iteration " << iteration_count << " timing: "
            << " Processing=" << total_processing_time / iteration_count
            << "s, Wait=" << wait_time 
            << "s (" << (wait_time > 0.001 ? "IO bound" : "CPU bound") << ")" << endl;
     }
   }
  
   // 最后处理剩余的重叠部分
   if (overlap > 0)
   {
     if (params.verbosity >= 1)
       cout << "Final sub gulp: overlap=" << overlap << endl;
       
     hd_size nsamps_processed;
     hd_size nsamps_to_process = overlap;
     if (nsamps_to_process > nsamps_gulp)
       nsamps_to_process = nsamps_gulp;
     
     auto process_start = std::chrono::high_resolution_clock::now();
     error = hd_execute(pipeline, &filterbank[0], nsamps_to_process, nbits, 
                        total_nsamps, &nsamps_processed);
     total_processing_time += get_time_diff(process_start);
     
     if (params.verbosity >= 1)
       cout << "Final sub gulp: nsamps_processed=" << nsamps_processed << endl;
 
     if (error == HD_NO_ERROR)
     { 
       if (params.verbosity >= 1)
         cout << "Processed " << nsamps_processed << " samples." << endl;
     }
     else if (error == HD_TOO_MANY_EVENTS)
     { 
       if (params.verbosity >= 1)
         cerr << "WARNING: hd_execute produces too many events, some data skipped" << endl;
     }
     else if (error == HD_TOO_FEW_NSAMPS)
     {
       if (params.verbosity >= 1)
         cerr << "WARNING: hd_execute did not have enough samples to process" << endl;
     }
     else
     {
       cerr << "ERROR: Pipeline execution failed" << endl;
       cerr << "       " << hd_get_error_string(error) << endl;
     }
     total_nsamps += nsamps_processed;
   }
   
   auto pipeline_e = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> pipeline_time = pipeline_e - pipeline_s;
   cout<< "pipeline time: "<< pipeline_time.count() << endl;
   
   // 输出总体时间统计
   cout << "===== 异步IO性能统计 =====" << endl;
   cout << "总处理时间: " << total_processing_time << " 秒" << endl;
   cout << "总IO等待时间: " << total_wait_time << " 秒" << endl;
   cout << "初始IO时间: " << total_io_time << " 秒" << endl;
   cout << "IO重叠率: " << (1.0 - (total_wait_time / total_processing_time)) * 100 << "%" << endl;
   cout << "理论非异步总时间: " << (total_processing_time + total_io_time + (iteration_count-1) * (total_wait_time / iteration_count)) << " 秒" << endl;
   cout << "实际总时间: " << pipeline_time.count() << " 秒" << endl;
   cout << "估计节省时间: " << (total_processing_time + total_io_time + (iteration_count-1) * (total_wait_time / iteration_count)) - pipeline_time.count() << " 秒" << endl;
   cout << "=========================" << endl;
    
   if( params.verbosity >= 1 ) {
     cout << "Successfully processed a total of " << total_nsamps
          << " samples." << endl;
   }
     
   if( params.verbosity >= 1 ) {
     cout << "Shutting down..." << endl;
   }
   
   hd_destroy_pipeline(pipeline);
   
   if( params.verbosity >= 1 ) {
     cout << "All done." << endl;
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> elapsed = end - start;
   cout<<"Total time taken: "<<elapsed.count()<<" seconds"<<endl;
 }