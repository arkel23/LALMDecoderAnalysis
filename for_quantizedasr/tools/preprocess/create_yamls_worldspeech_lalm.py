import os
import csv
import sys

import yaml

# The registry below needs the study's own facts (which cell a config belongs to, which split a
# checkpoint was selected on). Those live in the analysis repo's utils.py, one directory up from
# for_quantizedasr/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from utils import (to_study_cell, is_selection_split, in_domain_role,  # noqa: E402
                   SELECTION_SPLIT, TRAIN_CONFIGS, MULTI_CONFIG_TRAIN)


class QuotedStr(str):
    pass


def quoted_scalar(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")


yaml.add_representer(QuotedStr, quoted_scalar)

# Every WorldSpeech config, copied from configs/train/worldspeech_llama_questions.yaml.
configs = [
    "af_za", "am_et", "ar_bh", "ar_dz", "ar_eg", "ar_iq", "ar_kw", "ar_ma", "ar_sa", "ar_tn",
    "ar_un", "as_in", "az_az", "be_by", "bn_bd", "bn_in", "ca_es", "ca_fr", "ckb_iq", "cnr_me",
    "crs_sc", "cs_cz", "de_at", "de_li", "dgo_in", "dv_mv", "el_cy", "el_gr", "en_au", "en_jm",
    "en_ke", "en_nz", "en_pk", "en_sl", "en_us", "en_zm", "eo", "es_ar", "es_cl", "es_co",
    "es_es", "es_mx", "es_pe", "es_pr", "es_py", "es_uy", "fa_ir", "fr_ca", "fr_cd", "fr_ci",
    "grc_gr", "gu_in", "ha_ng", "ha_td", "he_il", "hi_in", "hu_hu", "hy_am", "id_id", "ig_ng",
    "ja_jp", "ka_ge", "kk_kz", "kn_in", "ko_kr", "kok_in", "la_va", "lb_lu", "mai_in", "mfe_mu",
    "mi_nz", "ml_in", "mn_mn", "mr_in", "ms_my", "ne_in", "ne_np", "nl_be", "nl_nl", "nr_za",
    "nso_za", "om_et", "or_in", "pa_in", "pl_pl", "pt_br", "rm_ch", "ro_md", "ro_ro", "ru_by",
    "ru_ru", "rw_rw", "si_lk", "sm_ws", "sn_zw", "sq_al", "sq_xk", "ss_za", "st_za", "sv_ax",
    "sw_ke", "sw_tz", "ta_in", "ta_lk", "te_in", "th_th", "ti_et", "tl_ph", "tn_bw", "tn_za",
    "tr_tr", "ts_za", "ur_in", "ur_pk", "uz_uz", "ve_za", "xh_za", "yue_hk", "zh_tw", "zu_za",
]

# crs_sc uses the ERISLab mirror, whose splits drop samples whose decoded audio length
# disagrees with the corpus `duration` column. disco-eth's crs_sc is uncleaned.
overrides = {"crs_sc": ("ERISLab/WorldSpeech", "test_clean")}

output_dir = "configs/datasets/short_ml"
os.makedirs(output_dir, exist_ok=True)

for config in configs:
    dataset_path, split = overrides.get(config, ("disco-eth/WorldSpeech", "test"))

    yaml_data = {
        "dataset_path": QuotedStr(dataset_path),
        "dataset": QuotedStr(config),
        "split": QuotedStr(split),
        "force_asr_language": QuotedStr(config.split("_")[0]),
        "eval_metrics": ["wer_all", "cer"],
    }

    filename = f"worldspeech_{config}_test.yaml"
    with open(os.path.join(output_dir, filename), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {filename}")

print(f"\nGenerated {len(configs)} config files in {output_dir}.")

# --- The eval-dataset registry -----------------------------------------------------------
# One row per eval config, and the single source of truth for which datasets the sweeps run.
# This is a second job for a config generator, which the conventions warn against -- but the
# alternative is a fourth place the language list is typed, and it has already drifted once:
# the baselines sweep had 43 datasets and the trained sweep 44, differing on exactly the config
# that must not be evaluated.
#
# use_in_sweep is DERIVED. worldspeech_ha_ng_test is Hausa's training-time selection split, so
# evaluating a Hausa checkpoint on it is not a held-out number. Nothing here is hand-excluded.

FLEURS_CELLS = ["am_et", "en_us", "es_419", "fr_fr", "ha_ng", "hi_in",
                "id_id", "mr_in", "sw_ke", "ta_in", "ur_pk"]

# The WorldSpeech configs each cell actually trained on.
trained_on = set()
for cell, entry in TRAIN_CONFIGS.items():
    names = entry[1]
    trained_on.update(names if isinstance(names, tuple) else (names,))

STUDY_PREFIXES = {c.split("_")[0] for c in FLEURS_CELLS} | {"crs"}

rows = []
for cell in FLEURS_CELLS:
    rows.append({"config_yaml": f"short_ml/fleurs_{cell}_test.yaml",
                 "dataset_path": "google/fleurs", "dataset": cell, "split": "test",
                 "source": "fleurs"})
for config in configs:
    if config.split("_")[0] not in STUDY_PREFIXES:
        continue
    dataset_path, split = overrides.get(config, ("disco-eth/WorldSpeech", "test"))
    rows.append({"config_yaml": f"short_ml/worldspeech_{config}_test.yaml",
                 "dataset_path": dataset_path, "dataset": config, "split": split,
                 "source": "worldspeech"})

for r in rows:
    cell = to_study_cell(r["dataset"])
    r["study_cell"] = cell
    r["in_training"] = r["source"] == "worldspeech" and r["dataset"] in trained_on
    r["is_selection_split"] = is_selection_split(
        cell, r["dataset_path"], r["dataset"], r["split"])
    r["in_domain_role"] = in_domain_role(
        cell, r["dataset"], "in_domain" if r["source"] == "worldspeech" else "cross_domain")
    r["use_in_sweep"] = not r["is_selection_split"]

registry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_datasets.csv")
with open(registry, "w", newline="") as f:
    # lineterminator: csv defaults to CRLF, which leaves a trailing \r on the last field and
    # makes the shell readers below match nothing.
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

n_out = sum(1 for r in rows if not r["use_in_sweep"])
print(f"Wrote {len(rows)} eval datasets to {registry} "
      f"({len(rows) - n_out} swept, {n_out} excluded as a selection split).")
for r in rows:
    if not r["use_in_sweep"]:
        print(f"  excluded: {r['config_yaml']} -- it is {r['study_cell']}'s selection split")
