script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd -P )
# script_dir=$( dirname -- "$( readlink -f -- "${BASH_SOURCE[0]}" )" )
cd "$script_dir"

aws s3 sync \
    s3://yiming-qu/20240227_yiwei_eb_batch2/ \
    . \
    --exclude "*" \
    --include "fastq/*" \
    --include "metadata/*" \
    --include "resolve.ipynb"
