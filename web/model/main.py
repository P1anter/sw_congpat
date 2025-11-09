# main.py (sw/web/model/ 디렉토리에 위치)

import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

# FastAPI 라이브러리 임포트
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# --- 1. 프로젝트 내부 모듈 (test_v5.py와 동일) ---
# (main.py와 p_dataloader_5_3.py가 같은 폴더에 있으므로 오류 없이 작동)
from p_dataloader_5_3 import (
    HadmTableDatasetV3, collate_hadm_batch_v3, example_sources_config_v3
)
from architectures.predictor.predict_modelv2 import TableTransformerPredictor

# --- 2. 경로 설정 (오류 방지를 위해 절대 경로로 변환) ---
# 현재 파일(main.py)의 위치를 기준으로 삼습니다. (sw/web/model/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# test_v5.py의 상대 경로를 BASE_DIR 기준으로 절대 경로화합니다.
PATIENTS_CSV = os.path.join(BASE_DIR, "./filtered/patients2.csv")
UNIFIED_CSV = os.path.join(BASE_DIR, "../dataset/summarized_with_readmit30_test.csv")
CKPT_PATH   = os.path.join(BASE_DIR, "./checkpoints_exp_final/highperf_best_exp_final.pt")
CACHE_DIR   = os.path.join(BASE_DIR, "./latent_cache_v2_exp_final")

USE_EXAMPLE_SOURCES = True
SOURCES_JSON_PATH   = None
BATCH_SIZE = 64

if torch.cuda.is_available():
    ENCODE_DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    ENCODE_DEVICE = "mps"
else:
    ENCODE_DEVICE = "cpu"

# --- ITEM ID와 한국어 검사명 매핑 ---
ITEMID_MAPPING = {
    50868: "탄산수소(Bicarb)",  # 50868은 CO2 (Bicarb)일 확률이 높음
    50882: "탄산수소(Bicarb)",  # 50882 역시 CO2 (Bicarb)일 확률이 높음
    50893: "클로라이드(Chloride)", # 50893은 Chloride일 확률이 높음
    50902: "칼슘(Calcium)",    # 50902는 Calcium일 확률이 높음
    50912: "크레아티닌(Creatinine)", # 50912는 Creatinine일 확률이 높음
    50931: "포도당(Glucose)",    # 50931은 Glucose일 확률이 높음
    50971: "칼륨(Potassium)", 
    50983: "나트륨(Sodium)",
    51006: "혈중요소질소(BUN)",
    # 필요한 다른 itemid가 있다면 여기에 추가
}

# --- 3. 핵심 로직 (test_v5.py에서 그대로 복사) ---

# =========================
# Utilities
# =========================
def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_col(df: pd.DataFrame, candidates):
    """대소문자 무시하고 후보 중 존재하는 첫 컬럼명을 반환"""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

# ----------------------------------------------------------------------------
# 3. 핵심 로직: 새 함수 추가
# ----------------------------------------------------------------------------

def lookup_lab_tests(subject_id_str: str, unified_csv_path: str):
    """특정 환자의 검사 기록을 UNIFIED_CSV에서 조회합니다."""
    try:
        # UNIFIED_CSV (summarized_with_readmit30_test.csv) 파일을 읽습니다.
        df_full = pd.read_csv(unified_csv_path) 
    except FileNotFoundError:
        return "❌ 오류: 통합 데이터 파일 경로가 존재하지 않습니다.", None
    
    try:
        subject_id = int(subject_id_str)
    except ValueError:
        return "❌ 오류: subject_id가 유효하지 않은 형식입니다.", None
    
    # subject_id로 필터링
    df_filtered = df_full[df_full['subject_id'] == subject_id]
    
    if df_filtered.empty:
        return "⚠️ 경고: 해당 환자(subject_id)의 검사 기록을 찾을 수 없습니다.", None
    
    # 웹에 표시할 검사 관련 컬럼만 선택
    # 수정: 'hadm_id'와 'days_of_visit'를 제거했습니다.
    lab_cols = [
        'charttime', 'itemid', 'valuenum', 'valueuom', 'value'
    ]
    # 중복된 검사 결과를 제거하고, 최근 기록 순으로 정렬 (hadm_id와 charttime 기준)
    # 정렬 기준은 'charttime'만 남겨도 되지만, 필터링 로직 유지를 위해 hadm_id를 사용하지 않더라도 기존 로직을 최소한으로 수정했습니다.
    df_tests = df_filtered[lab_cols + ['hadm_id']].drop_duplicates(
        subset=['hadm_id', 'charttime', 'itemid', 'valuenum'], keep='first'
    ).sort_values(by=['hadm_id', 'charttime'], ascending=False)
    
    # 최종적으로 반환할 DataFrame에서는 'hadm_id'를 제외합니다.
    df_tests = df_tests[lab_cols]
    
    info = f"✅ subject_id={subject_id}의 총 {len(df_tests)}건의 검사 기록 조회 완료."
    return info, df_tests

# =========================
# 1) 환자 조회 (JSON 반환을 위해 컬럼명 수정)
# =========================
def lookup_patient(subject_id_str: str, patients_csv_path: str):
    if not subject_id_str or not subject_id_str.strip().isdigit():
        return "❌ subject_id는 정수여야 합니다.", None
    sid = int(subject_id_str.strip())

    # 경로를 절대 경로로 변환 (FastAPI는 실행 위치가 다를 수 있으므로)
    if not os.path.isabs(patients_csv_path):
         patients_csv_path = os.path.join(BASE_DIR, patients_csv_path)

    if not os.path.isfile(patients_csv_path):
        return f"❌ patients.csv 경로가 존재하지 않습니다: {patients_csv_path}", None

    df = pd.read_csv(patients_csv_path, low_memory=False)

    c_subj   = _pick_col(df, ["subject_id"])
    c_name   = _pick_col(df, ["create", "name", "patient_name"])  # 'create'가 기본, 없으면 fallback
    c_gender = _pick_col(df, ["gender"])
    c_age    = _pick_col(df, ["anchor_age", "age"])
    c_year   = _pick_col(df, ["anchor_year", "birth_year"])

    for req, cname in {
        "subject_id": c_subj, "gender": c_gender, "anchor_age": c_age, "anchor_year": c_year
    }.items():
        if cname is None:
            return f"❌ patients.csv에 필요한 컬럼이 없습니다: {req}", None
    if c_name is None:
        # 이름 컬럼이 아예 없으면 빈 문자열로 대체
        df["__name__"] = "김지헌"
        c_name = "__name__"

    # 숫자 변환 후 필터
    df[c_subj] = pd.to_numeric(df[c_subj], errors="coerce").astype("Int64")
    sel = df.loc[df[c_subj] == sid, [c_subj, c_name, c_gender, c_age, c_year]].copy()

    if sel.empty:
        return f"⚠️ subject_id={sid} 에 해당하는 환자 정보가 없습니다.", None

    sel = sel.drop_duplicates().reset_index(drop=True)
    
    # 💥 중요: Gradio와 달리 JSON은 영문 key를 사용해야 합니다.
    sel.columns = ["subject_id", "patient_name", "gender", "anchor_age", "anchor_year"]

    info = f"✅ 환자 조회 완료: subject_id={sid} (행 {len(sel)}개)"
    return info, sel


# =========================
# 2) 퇴원일 예측
# =========================
@torch.no_grad()
def run_inference(
    subject_id_str: str,
    unified_csv: str,
    ckpt_path: str,
    use_example_sources: bool,
    sources_json_path: str,
    cache_dir: str,
    encode_device_str: str,  # "cpu" | "cuda" | "mps"
    batch_size: int,
):
    # 입력 검증
    if not subject_id_str or not subject_id_str.strip().isdigit():
        return "❌ subject_id는 정수여야 합니다.", None
    subject_id = int(subject_id_str.strip())
    
    # 경로 절대 경로로 변환
    if not os.path.isabs(unified_csv):
        unified_csv = os.path.join(BASE_DIR, unified_csv)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, ckpt_path)
    if cache_dir and not os.path.isabs(cache_dir):
        cache_dir = os.path.join(BASE_DIR, cache_dir)

    if not os.path.isfile(unified_csv):
        return f"❌ unified_csv 경로가 존재하지 않습니다: {unified_csv}", None
    if not os.path.isfile(ckpt_path):
        return f"❌ 체크포인트(.pt) 경로가 존재하지 않습니다: {ckpt_path}", None

    # sources 구성
    if use_example_sources:
        sources = example_sources_config_v3()
    else:
        if not sources_json_path or not os.path.isfile(sources_json_path):
            return "❌ sources_json_path가 유효하지 않습니다. 파일 경로를 확인하세요.", None
        with open(sources_json_path, "r", encoding="utf-8") as f:
            sources = json.load(f)

    # (선택) BRITS CKPT hidden_dim 미스매치 보정(필요시)
    if "complete_blood_count" in sources:
        sources["complete_blood_count"]["hidden_dim"] = sources["complete_blood_count"].get("hidden_dim", 64)

    # 단일 DS 생성(전체) → Subset으로 subject만 추출
    ds_all = HadmTableDatasetV3(
        unified_csv=unified_csv,
        drop_index_death=False,
        drop_30d_postdischarge_death=False,
        crrt_label_csv=None,
        mimic_derived_lods_csv=None,
        mimic_derived_crrt_csv=None,
        crrt_restrict_within_admission_window=True,
        sources=sources,
        encode_device=encode_device_str if encode_device_str in ("cpu","cuda","mps") else "cpu",
        cache_dir=cache_dir if (cache_dir and len(cache_dir.strip())>0) else None,
    )

    idxs = ds_all.df.index[ds_all.df["subject_id"] == subject_id].tolist()
    if len(idxs) == 0:
        return f"⚠️ subject_id={subject_id} 에 해당하는 HADM이 없습니다.", None

    subset = Subset(ds_all, idxs)

    # hadm → (admit, disch)
    tmp = ds_all.df.loc[idxs, ["hadm_id", "admittime", "dischtime"]].copy()
    tmp["admittime"] = pd.to_datetime(tmp["admittime"], errors="coerce")
    tmp["dischtime"] = pd.to_datetime(tmp["dischtime"], errors="coerce")
    times_map = {int(r.hadm_id): (r.admittime, r.dischtime) for r in tmp.itertuples(index=False)}

    # Loader
    loader = DataLoader(
        subset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_hadm_batch_v3,
        drop_last=False,
    )

    # peek → (S,N,E)
    peek = next(iter(loader))
    S = peek["base"].shape[1]
    N = peek["exam_z"].shape[1]
    E = peek["exam_z"].shape[2] if N > 0 else 0

    # 모델 로드
    device = resolve_device(None)
    model = TableTransformerPredictor(
        num_tables=N, latent_dim=E, base_dim=S,
        d_model=256, nhead=8, depth=3, dim_ff=768,
        dropout=0.15, head_hidden=256,
        use_film=True, use_masked_mean=True
    ).to(device).eval()

    ck = torch.load(ckpt_path, map_location="cpu")
    state_dict = ck.get("model", ck)  # {"model": ...} 또는 state_dict
    model.load_state_dict(state_dict, strict=False)

    # 추론
    rows, seen = [], set()
    for batch in loader:
        base = batch["base"].to(device)
        exam_z = batch["exam_z"].to(device)
        exam_mask = batch["exam_mask"].to(device)

        los_pred, readmit_logit = model(base=base, exam_z=exam_z, exam_mask=exam_mask)
        # readmit_prob = torch.sigmoid(readmit_logit)  # 필요시 사용

        hadm_list = batch["hadm_id"]
        subj_list = batch["subject_id"]
        for i in range(len(hadm_list)):
            hadm = int(hadm_list[i])
            subj = int(subj_list[i])

            lp = float(los_pred[i].item())
            lp_3 = round(lp, 3)  # 소수점 3자리

            # 실제 입/퇴원일
            admit_dt, disch_true_dt = times_map.get(hadm, (None, None))

            # 예측 퇴원일(LOS 3자리 사용)
            pred_disch_dt = None
            if admit_dt is not None and pd.notna(admit_dt):
                try:
                    pred_disch_dt = admit_dt + pd.to_timedelta(lp_3, unit="D")
                except Exception:
                    pred_disch_dt = None
            
            # 오차(일) → 소수점 3자리
            err_days_3 = None
            if (pred_disch_dt is not None) and (disch_true_dt is not None) and pd.notna(disch_true_dt):
                try:
                    err_days = (pred_disch_dt - disch_true_dt).total_seconds() / (24 * 3600.0)
                    err_days_3 = round(err_days, 3)
                except Exception:
                    err_days_3 = None

            key = (subj, hadm)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "subject_id": subj,
                "hadm_id": hadm,
                "admittime": admit_dt,
                "dischtime_true": disch_true_dt,
                "pred_dischtime": pred_disch_dt,  # ← lp_3로 계산된 datetime
                "los_pred_days": lp_3,            # ← 소수점 3자리
                "error_days": (float(err_days_3) if err_days_3 is not None else None),  # ← 소수점 3자리
            })

    df = pd.DataFrame(
        rows,
        columns=[
            "subject_id", "hadm_id", "admittime", "dischtime_true",
            "pred_dischtime", "los_pred_days", "error_days"
        ],
    ).drop_duplicates().reset_index(drop=True)

    # 날짜 보기 좋게
    for c in ["admittime", "dischtime_true", "pred_dischtime"]:
        if c in df.columns:
            # JSON 반환을 위해 NaT (Not a Time) 값을 None으로 변경
            df[c] = pd.to_datetime(df[c], errors='coerce')
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S").replace({pd.NaT: None})


    # 숫자 3자리 반올림 (표시는 float로 유지)
    for c in ["los_pred_days", "error_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    info = (
        f"✅ 예측 완료: subject_id={subject_id}, HADM {len(df)}건\n"
        f" - 모델 체크포인트: {os.path.basename(ckpt_path)}\n"
        f" - 테이블 수 N={N}, 잠재 차원 E={E}, base 차원 S={S}"
    )
    return info, df


# subject_id만 받아 예측 실행 (고정 인자 래핑)
def run_inference_defaults(subject_id_str: str):
    # 수정된 절대 경로 변수(UNIFIED_CSV, CKPT_PATH 등)를 사용합니다.
    return run_inference(
        subject_id_str=subject_id_str,
        unified_csv=UNIFIED_CSV,
        ckpt_path=CKPT_PATH,
        use_example_sources=USE_EXAMPLE_SOURCES,
        sources_json_path=SOURCES_JSON_PATH,
        cache_dir=CACHE_DIR,
        encode_device_str=ENCODE_DEVICE,
        batch_size=BATCH_SIZE,
    )


# --- 4. FastAPI 앱 설정 ---
app = FastAPI(title="Patient Prediction API")

# (선택사항) CORS 설정: 프론트엔드와 백엔드 포트가 다를 경우를 대비해 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 출처 허용 (테스트용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. API 엔드포인트(URL) 정의 ---

@app.get("/api/patient/{subject_id}")
async def get_patient_info(subject_id: str):
    """
    환자 ID를 받아 환자 기본 정보를 JSON으로 반환합니다.
    """
    # 1. lookup_patient 함수 실행 (절대 경로 PATIENTS_CSV 사용)
    info, sel = lookup_patient(subject_id, PATIENTS_CSV) 
    
    # 2. 오류 처리 (sel이 None이거나 비어있을 때)
    if sel is None or sel.empty:
        # info 변수에 담긴 오류 메시지를 404/500 에러로 반환
        error_detail = info.replace("❌ ", "").replace("⚠️ ", "")
        if "경로가 존재하지 않습니다" in error_detail or "컬럼이 없습니다" in error_detail:
             raise HTTPException(status_code=500, detail=error_detail) # 서버 설정 오류
        else:
             raise HTTPException(status_code=404, detail=error_detail) # 데이터 없음
    
    # 3. 성공 시: DataFrame의 첫 번째 행을 JSON(dict)으로 변환하여 반환
    # .iloc[0].to_dict()는 NaT/NaN 값을 JSON이 처리 못할 수 있으므로, 
    # Pandas의 JSON 변환 기능을 사용합니다.
    result_json = json.loads(sel.iloc[[0]].to_json(orient="records"))[0]
    return result_json


@app.get("/api/predict/{subject_id}")
async def get_prediction(subject_id: str):
    """
    환자 ID를 받아 퇴원일 예측 결과를 JSON으로 반환합니다.
    """
    # 1. run_inference_defaults 함수 실행
    info, df_result = run_inference_defaults(subject_id)
    
    # 2. 오류 처리
    if df_result is None:
        error_detail = info.replace("❌ ", "").replace("⚠️ ", "")
        if "경로가 존재하지 않습니다" in error_detail:
            raise HTTPException(status_code=500, detail=error_detail) # 서버 설정 오류
        else:
             raise HTTPException(status_code=404, detail=error_detail) # 데이터 없음
    
    # 3. 성공 시: 전체 예측 결과 DataFrame을 JSON (List[dict])으로 반환
    return {
        "status_message": "✅ 예측완료", #info 상세정보 사용시 info
        "predictions": json.loads(df_result.to_json(orient="records"))
    }

# ----------------------------------------------------------------------------
# 5. API 엔드포인트(URL) 정의: 새 엔드포인트 추가
# ----------------------------------------------------------------------------

@app.get("/api/tests/{subject_id}")
async def get_lab_tests(subject_id: str):
    """
    환자 ID를 받아 해당 환자의 임상 검사 기록을 반환합니다.
    """
    info, df_result = lookup_lab_tests(subject_id, UNIFIED_CSV)
    
    if df_result is None:
        error_detail = info.replace("❌ ", "").replace("⚠️ ", "")
        status_code = 500 if "경로가 존재하지 않습니다" in error_detail or "찾을 수 없습니다" in error_detail else 404
        raise HTTPException(status_code=status_code, detail=error_detail)
    
    # JSON 형식으로 반환
    return {
        "status_message": info,
        "lab_tests": json.loads(df_result.to_json(orient="records")),
        "item_id_map": ITEMID_MAPPING # 매핑 정보 추가
    }

# --- 6. 프론트엔드 파일 서빙 (404 오류 해결) ---
FRONTEND_DIR = os.path.join(BASE_DIR, "../../frontend") # sw/frontend 경로

# 1) 루트 경로 '/' 요청 시 index.html을 서빙합니다.
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Error: index.html not found. Check if the 'frontend' folder is at the correct path.</h1>", status_code=500)
    with open(index_path, 'r', encoding='utf-8') as f:
        return f.read()

# 2) 정적 파일 (CSS, JS) 요청 시 '/static' 경로로 마운트합니다.
# 이제 index.html에서 <link rel="stylesheet" href="/static/style.css"> 로 파일을 제대로 찾습니다.
app.mount(
    "/static", 
    StaticFiles(directory=FRONTEND_DIR), 
    name="static"
)

# --- 7. 서버 실행 (uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    # uvicorn main:app --reload --host 127.0.0.1 --port 8000
    print(f"FastAPI 서버를 시작합니다. (BASE_DIR: {BASE_DIR})")
    print(f"Frontend 경로: {FRONTEND_DIR}")
    print("터미널에서 'uvicorn main:app --reload --host 127.0.0.1 --port 8000' 명령을 실행하세요.")
    uvicorn.run(app, host="127.0.0.1", port=8000)