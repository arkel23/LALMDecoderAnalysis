import os
import yaml


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

# Non-space-delimited scripts, matching create_yamls_fleurs_full.py's CER_LANG_IDS.
cer_configs = {"ja_jp", "th_th", "yue_hk", "zh_tw"}

output_dir = "configs/datasets/short_ml"
os.makedirs(output_dir, exist_ok=True)

for config in configs:
    dataset_path, split = overrides.get(config, ("disco-eth/WorldSpeech", "test"))

    yaml_data = {
        "dataset_path": QuotedStr(dataset_path),
        "dataset": QuotedStr(config),
        "split": QuotedStr(split),
        "force_asr_language": QuotedStr(config.split("_")[0]),
        "eval_metrics": ["cer"] if config in cer_configs else ["wer_all"],
    }

    filename = f"worldspeech_{config}_test.yaml"
    with open(os.path.join(output_dir, filename), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {filename}")

print(f"\nGenerated {len(configs)} config files in {output_dir}.")
