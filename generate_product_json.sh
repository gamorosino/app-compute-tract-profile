#!/usr/bin/env bash
set -euo pipefail

datatype_tags=()

function join_by { local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}"; }

datatype_tags_str=""
if [ ${#datatype_tags[@]} -gt 0 ]; then
    datatype_tags_str=$(join_by , "${datatype_tags[@]}")
fi

qa_entries=()

echo "==== DEBUG: scanning for images ===="
while IFS= read -r image; do
    echo "Processing image: $image"

    base=$(basename "$image" .png)

    entry=$(printf '{"type":"image/png","name":"%s","base64":"%s"}' \
        "$base" \
        "$(base64 -w 0 "$image")")

    qa_entries+=("$entry")
done < <(find "${1}" -name "*.png" | sort)

echo "==== DEBUG: number of entries: ${#qa_entries[@]} ===="

if [ ${#qa_entries[@]} -eq 0 ]; then
    brainlife_json='{"type":"error","msg":"Failed to generate output image."}'
else
    brainlife_json=$(printf '%s\n' "${qa_entries[@]}" | paste -sd, -)
fi

echo "==== DEBUG: final JSON chunk ===="
echo "$brainlife_json"

cat << EOF > "${2}"
{
  "datatype_tags": [],
  "brainlife": [${brainlife_json}]
}
EOF
