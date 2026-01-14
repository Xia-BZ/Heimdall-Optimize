After Compiling a command-line applicaiton named heimdall will be built, as well as some other utiliities. The command line options for the heimdall executable can be displayed by running heimdall -h and is shown below:

Usage: heimdall [options]
-k  key                  use PSRDADA hexidecimal key
-f  filename             process specified SIGPROC filterbank file
-vVgG                    increase verbosity level
-yield_cpu               yield CPU during GPU operations
-gpu_id ID               run on specified GPU
-nsamps_gulp num         number of samples to be read at a time [262144]
-baseline_length num     number of seconds over which to smooth the baseline [2]
-beam ##                 over-ride beam number
-output_dir path         create all output files in specified path
-dm min max              min and max DM
-dm_tol num              SNR loss tolerance between each DM trial [1.25]
-coincidencer host:port  connect to the coincidencer on the specified host and port
-zap_chans start end     zap all channels between start and end channels inclusive
-max_giant_rate nevents  limit the maximum number of individual detections per minute to nevents
-dm_pulse_width num      expected intrinsic width of the pulse signal in microseconds
-dm_nbits num            number of bits per sample in dedispersed time series [32]
-no_scrunching           don't use an adaptive time scrunching during dedispersion
-scrunching_tol num      smear tolerance factor for time scrunching [1.15]
-rfi_tol num             RFI exicision threshold limits [5]
-rfi_no_narrow           disable narrow band RFI excision
-rfi_no_broad            disable 0-DM RFI excision
-boxcar_max num          maximum boxcar width in samples [4096]
-fswap                   swap channel ordering for negative DM - SIGPROC 2,4 or 8 bit only
-min_tscrunch_width num  vary between high quality (large value) and high performance (low value)
