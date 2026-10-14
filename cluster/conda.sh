# Sourced by the cluster scripts: load conda (Miniconda, or Miniforge if that's what you have)
# and activate the project env.  Usage: source cluster/conda.sh
for _d in "$HOME/miniconda3" "$HOME/miniforge3"; do
  if [ -f "$_d/etc/profile.d/conda.sh" ]; then
    source "$_d/etc/profile.d/conda.sh"
    break
  fi
done
command -v conda >/dev/null || { echo "conda not found: run cluster/setup_env.sh first" >&2; exit 1; }
case $- in *u*) _had_u=1; set +u ;; *) _had_u=0 ;; esac  # conda activate trips over `set -u`
conda activate "${QA_ENV:-qa}"
[ "$_had_u" = 1 ] && set -u
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
