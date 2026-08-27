#!/usr/bin/env bash
# Shared matrix enumerator. Call run_experiment_matrix with a launcher-tree path.

set -euo pipefail

matrix_contains() {
    local csv="${1// /,}"
    local wanted="$2"
    case ",$csv," in
        *,"$wanted",*) return 0 ;;
        *) return 1 ;;
    esac
}

run_experiment_matrix() {
    local launcher_root="$1"
    shift
    local dry_run="${DRY_RUN:-0}"
    local families="${MATRIX_FAMILIES:-baseline,opd,opsd}"
    local models="${MATRIX_MODELS:-1B,4B,8B}"
    local variants="${MATRIX_VARIANTS:-}"
    local continue_on_error="${MATRIX_CONTINUE_ON_ERROR:-0}"
    local -a forwarded_args=()
    local arg

    for arg in "$@"; do
        case "$arg" in
            --dry-run) dry_run=1 ;;
            *) forwarded_args+=("$arg") ;;
        esac
    done
    case "$dry_run" in
        1|true|TRUE|yes|YES) dry_run=1 ;;
        0|false|FALSE|no|NO) dry_run=0 ;;
        *) echo "ERROR: DRY_RUN must be 0/1 or false/true (got: $dry_run)" >&2; return 2 ;;
    esac
    case "$continue_on_error" in
        1|true|TRUE|yes|YES) continue_on_error=1 ;;
        0|false|FALSE|no|NO) continue_on_error=0 ;;
        *) echo "ERROR: MATRIX_CONTINUE_ON_ERROR must be 0/1 or false/true" >&2; return 2 ;;
    esac

    local -a family_dirs=(Baselines OPD OPSD)
    local -a baseline_variants=(base sft grpo)
    local -a distill_variants=(vanilla top_k clip skd trd)
    local family_dir family model variant launcher launcher_status
    local launched=0 failed=0
    for family_dir in "${family_dirs[@]}"; do
        case "$family_dir" in
            Baselines) family=baseline ;;
            OPD) family=opd ;;
            OPSD) family=opsd ;;
            *) echo "ERROR: unsupported matrix family: $family_dir" >&2; return 2 ;;
        esac
        matrix_contains "$families" "$family" || continue
        for model in 1B 4B 8B; do
            matrix_contains "$models" "$model" || continue
            local -a selected_variants=()
            if [ "$family" = baseline ]; then
                selected_variants=("${baseline_variants[@]}")
            else
                selected_variants=("${distill_variants[@]}")
            fi
            for variant in "${selected_variants[@]}"; do
                if [ -n "$variants" ] && ! matrix_contains "$variants" "$variant"; then
                    continue
                fi
                launcher="$launcher_root/$family_dir/$model/$variant.sh"
                if [ ! -f "$launcher" ]; then
                    echo "ERROR: matrix launcher is missing: $launcher" >&2
                    failed=$((failed + 1))
                    [ "$continue_on_error" = 1 ] && continue
                    return 1
                fi
                echo "[$((launched + 1))] $family/$model/$variant"
                if [ "${#forwarded_args[@]}" -gt 0 ]; then
                    launcher_status=0
                    DRY_RUN="$dry_run" bash "$launcher" "${forwarded_args[@]}" || launcher_status=$?
                else
                    launcher_status=0
                    DRY_RUN="$dry_run" bash "$launcher" || launcher_status=$?
                fi
                if [ "$launcher_status" -ne 0 ]; then
                    failed=$((failed + 1))
                    [ "$continue_on_error" = 1 ] || return 1
                fi
                launched=$((launched + 1))
            done
        done
    done
    if [ "$launched" -eq 0 ]; then
        echo "ERROR: matrix filters selected no experiments" >&2
        return 2
    fi
    echo "Matrix complete: launched=$launched failed=$failed dry_run=$dry_run"
    [ "$failed" -eq 0 ]
}
