Here's a clean and well-formatted Markdown version of your command-line help text, suitable for inclusion in a README.md file:

Usage

After compiling, a command-line application named heimdall will be built, along with several utility programs.

To display the available command-line options, run:

bash
heimdall -h

The full usage information is shown below:

text
Usage: heimdall [options]

Options
Option   Description
-k key   Use PSRDADA hexadecimal key.

-f filename   Process the specified SIGPROC filterbank file.

-v, -V, -g, -G   Increase verbosity level.

-yield_cpu   Yield CPU during GPU operations.

-gpu_id ID   Run on the specified GPU (by ID).

-nsamps_gulp num   Number of samples to read at a time. Default: 262144.

-baseline_length num   Number of seconds over which to smooth the baseline. Default: 2.

-beam ##   Override beam number.

-output_dir path   Create all output files in the specified directory.

-dm min max   Set minimum and maximum dispersion measure (DM) to search.

-dm_tol num   SNR loss tolerance between DM trials. Default: 1.25.

-coincidencer host:port   Connect to a coincidencer running on the specified host and port.

-zap_chans start end   Zap (exclude) all channels from start to end inclusive.

-max_giant_rate nevents   Limit the maximum number of individual detections per minute to nevents.

-dm_pulse_width num   Expected intrinsic pulse width (in microseconds).

-dm_nbits num   Number of bits per sample in the dedispersed time series. Default: 32.

-no_scrunching   Disable adaptive time scrunching during dedispersion.

-scrunching_tol num   Smear tolerance factor for time scrunching. Default: 1.15.

-rfi_tol num   RFI excision threshold limit. Default: 5.

-rfi_no_narrow   Disable narrowband RFI excision.

-rfi_no_broad   Disable 0-DM (broadband) RFI excision.

-boxcar_max num   Maximum boxcar width (in samples). Default: 4096.

-fswap   Swap channel ordering for negative DM (SIGPROC 2-, 4-, or 8-bit only).

-min_tscrunch_width num   Trade-off between quality (larger value) and performance (smaller value).

💡 Tip: Most numerical parameters have sensible defaults—only override them if you understand their impact on sensitivity or performance.
`
